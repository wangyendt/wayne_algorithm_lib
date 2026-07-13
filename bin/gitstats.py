#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import subprocess
import sys
import warnings
from typing import Optional, Tuple

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from matplotlib.ft2font import FT2Font
from matplotlib.ticker import MaxNLocator

from pywayne.tools import wayne_print


# Compact code-review dashboard: quiet neutrals, one Git-inspired accent, and a
# single-hue activity scale that stays perceptually consistent.
CANVAS = "#FBFCFD"
PANEL = "#FFFFFF"
INK = "#18232D"
MUTED = "#6B7B87"
GRID = "#E8EDF0"
ACCENT = "#E4573D"
ACCENT_DARK = "#BA3E2B"
STEEL = "#416C7D"
STEEL_LIGHT = "#AFC2CB"

FIGURE_SIZE = (11, 7.2)

CJK_FONT_CANDIDATES = (
    "PingFang SC",
    "Microsoft YaHei",
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Noto Sans CJK TC",
    "Noto Sans CJK HK",
    "Noto Sans SC",
    "Noto Sans TC",
    "Noto Sans JP",
    "Source Han Sans SC",
    "Source Han Sans CN",
    "Hiragino Sans GB",
    "WenQuanYi Micro Hei",
    "WenQuanYi Zen Hei",
    "Droid Sans Fallback",
    "AR PL UKai CN",
    "AR PL UMing CN",
    "Unifont",
    "SimHei",
    "Arial Unicode MS",
    "Heiti SC",
    "Heiti TC",
    "Songti SC",
)

CJK_TEST_TEXT = "中文提交分析趋势时段星期分布活跃最高次数日周月小时"

CHART_TEXT = {
    "zh": {
        "title": "Git 提交分析",
        "summary": "{commits:,} 次提交  ·  {days:,} 个活跃日",
        "trend": "提交趋势  ·  {period}",
        "periods": {"day": "按日", "week": "按周", "month": "按月"},
        "peak": "最高 {count}",
        "hour_title": "提交时段",
        "hour": "小时",
        "weekday_title": "星期分布",
        "weekdays": ["周一", "周二", "周三", "周四", "周五", "周六", "周日"],
        "heatmap_title": "活跃时段  ·  星期 × 小时",
    },
    "en": {
        "title": "Git Commit Analysis",
        "summary": "{commits:,} commits  ·  {days:,} active days",
        "trend": "Commit trend  ·  {period}",
        "periods": {"day": "Daily", "week": "Weekly", "month": "Monthly"},
        "peak": "Peak {count}",
        "hour_title": "Commits by hour",
        "hour": "Hour",
        "weekday_title": "Commits by weekday",
        "weekdays": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        "heatmap_title": "Activity  ·  Weekday x Hour",
    },
}

HEATMAP_CMAP = sns.light_palette(STEEL, as_cmap=True)


def _refresh_system_fonts() -> None:
    """Register fonts installed after Matplotlib's font cache was created."""
    registered = {
        os.path.realpath(entry.fname) for entry in font_manager.fontManager.ttflist
    }
    for path in font_manager.findSystemFonts():
        if os.path.realpath(path) in registered:
            continue
        try:
            font_manager.fontManager.addfont(path)
        except (OSError, RuntimeError, TypeError, ValueError):
            continue


def _font_supports_text(font_path: str, text: str) -> bool:
    """Return whether a font file contains every glyph needed by ``text``."""
    try:
        charmap = FT2Font(font_path).get_charmap()
    except (OSError, RuntimeError, TypeError, ValueError):
        return False
    return all(ord(character) in charmap for character in text)


def _find_cjk_font() -> Optional[str]:
    """Find a real CJK-capable font, preferring common platform families."""
    _refresh_system_fonts()
    entries = font_manager.fontManager.ttflist

    for candidate in CJK_FONT_CANDIDATES:
        for entry in entries:
            if entry.name.casefold() == candidate.casefold() and _font_supports_text(
                entry.fname, CJK_TEST_TEXT
            ):
                return entry.name

    for entry in entries:
        if _font_supports_text(entry.fname, CJK_TEST_TEXT):
            return entry.name
    return None


def configure_fonts() -> Optional[str]:
    """Configure a font stack after verifying that its CJK glyphs exist."""
    cjk_font = _find_cjk_font()

    font_stack = ["DejaVu Sans"]
    if cjk_font:
        font_stack.append(cjk_font)

    mpl.rcParams.update(
        {
            "font.family": font_stack,
            "axes.unicode_minus": False,
            "savefig.facecolor": CANVAS,
        }
    )
    return cjk_font


def _configure_theme() -> Optional[str]:
    """Apply the Seaborn theme before restoring the CJK-aware font stack."""
    sns.set_theme(
        context="notebook",
        style="whitegrid",
        rc={
            "figure.facecolor": CANVAS,
            "axes.facecolor": PANEL,
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "axes.linewidth": 0.8,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
        },
    )
    return configure_fonts()


def get_repo_name(repo: str) -> str:
    try:
        top = subprocess.check_output(
            ["git", "-C", repo, "rev-parse", "--show-toplevel"], text=True
        ).strip()
        return os.path.basename(top)
    except subprocess.CalledProcessError:
        return os.path.basename(os.path.abspath(repo))


def get_current_branch(repo: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", repo, "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
    except subprocess.CalledProcessError:
        return "HEAD"


def read_commit_times(
    repo: str,
    since: Optional[str],
    until: Optional[str],
    tz: str,
    branch: Optional[str],
    include_all: bool,
) -> pd.DatetimeIndex:
    cmd = ["git", "-C", repo, "log", "--pretty=%aI"]
    if include_all:
        cmd.append("--all")
    elif branch:
        # 指定分支，如 main 或 origin/main；默认不指定分支即当前 HEAD
        cmd.append(branch)
    if since:
        cmd += ["--since", since]
    if until:
        cmd += ["--until", until]
    out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).splitlines()
    if not out:
        return pd.DatetimeIndex([])
    dt = pd.to_datetime(out, utc=True, errors="coerce").dropna()
    return dt.tz_convert(tz).tz_localize(None)


def _prepare_trend(commits: pd.Series) -> Tuple[pd.Series, str]:
    """Select a readable aggregation level for the visible time span."""
    span_days = max(1, (commits.index.max() - commits.index.min()).days)
    if span_days <= 180:
        frequency, period = "D", "day"
    elif span_days <= 3 * 365:
        frequency, period = "W-MON", "week"
    else:
        frequency, period = "MS", "month"
    return commits.resample(frequency).sum(), period


def _ascii_safe(value: str) -> str:
    """Keep dynamic repository metadata readable without a Unicode font."""
    return value.encode("ascii", errors="backslashreplace").decode("ascii")


def _date_formatter(locator: mdates.AutoDateLocator) -> mdates.ConciseDateFormatter:
    """Create a locale-independent formatter using only numeric date fields."""
    formatter = mdates.ConciseDateFormatter(locator)
    formatter.formats = ["%Y", "%m", "%d", "%H:%M", "%H:%M", "%S"]
    formatter.zero_formats = ["", "%Y", "%m", "%m-%d", "%H:%M", "%H:%M"]
    formatter.offset_formats = [
        "",
        "%Y",
        "%Y-%m",
        "%Y-%m-%d",
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M",
    ]
    return formatter


def _style_panel(ax: plt.Axes, *, grid: bool = True) -> None:
    ax.set_facecolor(PANEL)
    ax.tick_params(axis="both", labelsize=8.5, length=0, pad=5)
    if grid:
        ax.grid(axis="y", visible=True)
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)
    else:
        ax.grid(False)
    sns.despine(ax=ax, left=True, bottom=True)


def _set_panel_title(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="semibold", pad=8)


def _mark_peak_bar(ax: plt.Axes, index: int, value: int) -> None:
    peak_bar = ax.patches[index]
    peak_bar.set_facecolor(ACCENT)
    ax.annotate(
        f"{value}",
        xy=(peak_bar.get_x() + peak_bar.get_width() / 2, peak_bar.get_height()),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        va="bottom",
        color=ACCENT_DARK,
        fontsize=8,
        fontweight="semibold",
    )


def create_commit_figure(
    ts: pd.DatetimeIndex, repo_name: str, branch_label: str, timezone: str
) -> plt.Figure:
    """Create a compact Seaborn-based dashboard from commit timestamps."""
    cjk_font = _configure_theme()
    language = "zh" if cjk_font else "en"
    text = CHART_TEXT[language]
    if cjk_font is None:
        warnings.warn(
            "No CJK-capable font was found; using English chart labels. "
            "On Ubuntu, install one with: sudo apt install fonts-noto-cjk",
            RuntimeWarning,
            stacklevel=2,
        )
        repo_name = _ascii_safe(repo_name)
        branch_label = _ascii_safe(branch_label)

    commits = pd.Series(1, index=ts).sort_index()
    daily = commits.resample("D").sum()
    trend, trend_period = _prepare_trend(commits)
    by_hour = (
        commits.groupby(commits.index.hour).size().reindex(range(24), fill_value=0)
    )
    by_dow = (
        commits.groupby(commits.index.dayofweek).size().reindex(range(7), fill_value=0)
    )
    heat = (
        commits.groupby([commits.index.dayofweek, commits.index.hour])
        .size()
        .unstack(fill_value=0)
        .reindex(index=range(7), columns=range(24), fill_value=0)
    )

    weekdays = text["weekdays"]
    total_commits = int(commits.sum())
    active_days = int((daily > 0).sum())
    date_range = f"{ts.min().date()} — {ts.max().date()}"

    fig = plt.figure(figsize=FIGURE_SIZE, facecolor=CANVAS, constrained_layout=True)
    grid = fig.add_gridspec(
        4,
        2,
        height_ratios=(0.22, 1.08, 0.92, 0.88),
        width_ratios=(1, 1),
    )
    layout_engine = fig.get_layout_engine() if hasattr(fig, "get_layout_engine") else None
    if layout_engine is not None:
        layout_engine.set(w_pad=0.035, h_pad=0.035, wspace=0.06, hspace=0.07)
    else:
        fig.set_constrained_layout_pads(
            w_pad=0.035, h_pad=0.035, wspace=0.06, hspace=0.07
        )

    header = fig.add_subplot(grid[0, :])
    header.set_axis_off()
    header.text(
        0,
        0.72,
        text["title"],
        fontsize=17,
        fontweight="bold",
        ha="left",
        va="center",
    )
    header.text(
        0,
        0.10,
        f"{repo_name}  /  {branch_label}  /  {timezone}",
        color=MUTED,
        fontsize=8.5,
        ha="left",
        va="center",
    )
    header.text(
        1,
        0.67,
        date_range,
        fontsize=9.5,
        fontweight="semibold",
        ha="right",
        va="center",
    )
    header.text(
        1,
        0.10,
        text["summary"].format(commits=total_commits, days=active_days),
        color=MUTED,
        fontsize=8.5,
        ha="right",
        va="center",
    )

    ax_trend = fig.add_subplot(grid[1, :])
    _style_panel(ax_trend)
    _set_panel_title(
        ax_trend,
        text["trend"].format(period=text["periods"][trend_period]),
    )
    ax_trend.plot(
        trend.index,
        trend.values,
        color=ACCENT,
        linewidth=1.8,
        solid_capstyle="round",
    )
    ax_trend.fill_between(
        trend.index, trend.values, color=ACCENT, alpha=0.09, linewidth=0
    )
    peak_period = trend.idxmax()
    peak_count = int(trend.max())
    ax_trend.scatter(
        [peak_period],
        [peak_count],
        s=25,
        color=ACCENT,
        edgecolor=PANEL,
        linewidth=1,
        zorder=4,
    )
    ax_trend.annotate(
        text["peak"].format(count=peak_count),
        xy=(peak_period, peak_count),
        xytext=(0, 7),
        textcoords="offset points",
        ha="center",
        color=ACCENT_DARK,
        fontsize=8,
        fontweight="semibold",
    )
    date_locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
    ax_trend.xaxis.set_major_locator(date_locator)
    ax_trend.xaxis.set_major_formatter(_date_formatter(date_locator))
    ax_trend.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax_trend.set(xlabel=None, ylabel=None)
    ax_trend.set_ylim(bottom=0, top=max(1, peak_count * 1.20))
    ax_trend.margins(x=0.006)

    ax_hour = fig.add_subplot(grid[2, 0])
    _style_panel(ax_hour)
    _set_panel_title(ax_hour, text["hour_title"])
    ax_hour.bar(
        range(24),
        by_hour.values,
        color=STEEL_LIGHT,
        width=0.72,
        edgecolor="none",
    )
    peak_hour = int(by_hour.values.argmax())
    _mark_peak_bar(ax_hour, peak_hour, int(by_hour.iloc[peak_hour]))
    hour_ticks = list(range(0, 24, 3))
    ax_hour.set_xticks(hour_ticks, labels=[f"{hour:02d}" for hour in hour_ticks])
    ax_hour.set(xlabel=text["hour"], ylabel=None)
    ax_hour.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax_hour.set_ylim(0, max(1, int(by_hour.max()) * 1.18))

    ax_weekday = fig.add_subplot(grid[2, 1])
    _style_panel(ax_weekday)
    _set_panel_title(ax_weekday, text["weekday_title"])
    ax_weekday.bar(
        range(7),
        by_dow.values,
        color=STEEL,
        width=0.62,
        edgecolor="none",
    )
    peak_weekday = int(by_dow.values.argmax())
    _mark_peak_bar(ax_weekday, peak_weekday, int(by_dow.iloc[peak_weekday]))
    ax_weekday.set_xticks(range(7), labels=weekdays)
    ax_weekday.set(xlabel=None, ylabel=None)
    ax_weekday.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
    ax_weekday.set_ylim(0, max(1, int(by_dow.max()) * 1.18))

    ax_heatmap = fig.add_subplot(grid[3, :])
    _style_panel(ax_heatmap, grid=False)
    _set_panel_title(ax_heatmap, text["heatmap_title"])
    heat.index = weekdays
    heat.columns = [f"{hour:02d}" for hour in range(24)]
    sns.heatmap(
        heat,
        ax=ax_heatmap,
        cmap=HEATMAP_CMAP,
        vmin=0,
        linewidths=0.45,
        linecolor=PANEL,
        xticklabels=2,
        yticklabels=True,
        cbar_kws={"shrink": 0.78, "pad": 0.012, "aspect": 18},
    )
    ax_heatmap.set(xlabel=text["hour"], ylabel=None)
    ax_heatmap.tick_params(axis="x", rotation=0)
    ax_heatmap.tick_params(axis="y", rotation=0)
    colorbar = ax_heatmap.collections[0].colorbar
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=7.5, length=0)
    colorbar.locator = MaxNLocator(nbins=4, integer=True)
    colorbar.update_ticks()

    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description="统计 Git 提交时间分布并输出图表")
    ap.add_argument("repo", nargs="?", default=".", help="git 仓库路径，默认当前目录")
    ap.add_argument("--since", default=None, help='起始，如 "2024-01-01" 或 "1 year ago"')
    ap.add_argument("--until", default=None, help='终止，如 "2025-08-10"')
    ap.add_argument("--tz", default="Asia/Shanghai", help='时区，如 "Asia/Shanghai"、"UTC"')
    ap.add_argument("--branch", default=None, help="只统计指定分支，如 main 或 origin/main")
    ap.add_argument("--all", action="store_true", help="统计所有分支（忽略 --branch）")
    ap.add_argument("--save", default="git_time_distribution.png", help="图片输出路径")
    ap.add_argument(
        "-p",
        "--show-plot",
        "--show_plot",
        action="store_true",
        help="弹窗展示图表（不保存文件）",
    )

    args = ap.parse_args()

    repo_name = get_repo_name(args.repo)
    branch_label = "ALL" if args.all else (args.branch or get_current_branch(args.repo))

    try:
        ts = read_commit_times(
            args.repo, args.since, args.until, args.tz, args.branch, args.all
        )
    except subprocess.CalledProcessError as error:
        sys.exit(f"[git 调用失败]\n{error}")
    if ts.empty:
        sys.exit("没有读到提交记录。检查仓库路径、分支名或时间过滤条件。")

    fig = create_commit_figure(ts, repo_name, branch_label, args.tz)

    if args.show_plot:
        plt.show()
    else:
        fig.savefig(args.save, dpi=160)
        plt.close(fig)
        wayne_print(f"Saved: {args.save}", "green")


if __name__ == "__main__":
    main()
