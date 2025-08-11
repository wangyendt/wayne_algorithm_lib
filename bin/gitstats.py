#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import subprocess
import sys
import os
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt

from pywayne.plot import parula_map
from pywayne.tools import wayne_print


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


def main() -> None:
    ap = argparse.ArgumentParser(description="统计 Git 提交时间分布并输出图表")
    ap.add_argument("repo", nargs="?", default=".", help="git 仓库路径，默认当前目录")
    ap.add_argument("--since", default=None, help='起始，如 "2024-01-01" 或 "1 year ago"')
    ap.add_argument("--until", default=None, help='终止，如 "2025-08-10"')
    ap.add_argument("--tz", default="Asia/Shanghai", help='时区，如 "Asia/Shanghai"、"UTC"')
    ap.add_argument("--branch", default=None, help="只统计指定分支，如 main 或 origin/main")
    ap.add_argument("--all", action="store_true", help="统计所有分支（忽略 --branch）")
    ap.add_argument("--save", default="git_time_distribution.png", help="图片输出路径")
    ap.add_argument("-p", "--show-plot", "--show_plot", action="store_true", help="弹窗展示图表（不保存文件）")

    args = ap.parse_args()

    repo_name = get_repo_name(args.repo)
    branch_label = "ALL" if args.all else (args.branch or get_current_branch(args.repo))

    try:
        ts = read_commit_times(
            args.repo, args.since, args.until, args.tz, args.branch, args.all
        )
    except subprocess.CalledProcessError as e:
        sys.exit(f"[git 调用失败]\n{e}")
    if ts.empty:
        sys.exit("没有读到提交记录。检查仓库路径、分支名或时间过滤条件。")

    s = pd.Series(1, index=ts).sort_index()

    daily = s.resample("D").sum()
    by_hour = s.groupby(s.index.hour).size().reindex(range(24), fill_value=0)
    by_dow = s.groupby(s.index.dayofweek).size().reindex(range(7), fill_value=0)
    heat = (
        s.groupby([s.index.dayofweek, s.index.hour])
        .size()
        .unstack(fill_value=0)
        .reindex(index=range(7), columns=range(24), fill_value=0)
    )

    weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    by_dow.index = weekdays

    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.15)

    # 顶部标题：项目名 + 分支 + 时区
    title = f"{repo_name} · {branch_label} · {args.tz}"
    fig.suptitle(f"Git Commit Time Distribution — {title}", y=0.98, fontsize=13)

    ax1 = fig.add_subplot(gs[0, :])
    daily.plot(ax=ax1)
    ax1.set_title("Commits per Day")
    ax1.set_ylabel("Commits")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 0])
    by_hour.plot(kind="bar", ax=ax2)
    ax2.set_title("Commits by Hour (0–23)")
    ax2.set_xlabel("Hour")
    ax2.set_ylabel("Commits")

    ax3 = fig.add_subplot(gs[1, 1])
    by_dow.plot(kind="bar", ax=ax3)
    ax3.set_title("Commits by Weekday")
    ax3.set_ylabel("Commits")

    ax4 = fig.add_subplot(gs[2, :])
    im = ax4.imshow(heat.values, aspect="auto", cmap=parula_map)
    ax4.set_title("Heatmap: Weekday × Hour")
    ax4.set_yticks(range(7), labels=weekdays)
    ax4.set_xlabel("Hour")
    cbar = fig.colorbar(im, ax=ax4)
    cbar.set_label("Commits")

    # 角落里补充时间范围与提交数
    rng = f"{ts.min().date()} → {ts.max().date()}"
    fig.text(0.01, 0.005, f"Range: {rng} | Commits: {int(s.sum())}", fontsize=9, alpha=0.8)

    fig.tight_layout()

    if args.show_plot:
        plt.show()
    else:
        fig.savefig(args.save, dpi=150)
        wayne_print(f"Saved: {args.save}", 'green')


if __name__ == "__main__":
    main()


