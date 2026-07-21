#!/usr/bin/env python3
"""Configure common developer commands and tools on macOS, Linux, and Windows."""

import argparse
import json
import os
import platform as platform_module
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse


TOOL_NAME = "toolsetup"
NVM_VERSION = "v0.40.6"
NODE_VERSION = "20"
NPM_REGISTRY = "https://registry.npmmirror.com"
MONGODB_VERSION = "8.0"
HTTP_PROXY = "http://127.0.0.1:7890"
SOCKS_PROXY = "socks5://127.0.0.1:7890"
NO_PROXY = "localhost,127.0.0.1,.qualcomm.com,*.amazonaws.com"

UNIX_BLOCK_START = "# >>> pywayne toolsetup shortcuts >>>"
UNIX_BLOCK_END = "# <<< pywayne toolsetup shortcuts <<<"
POWERSHELL_BLOCK_START = "# >>> pywayne toolsetup shortcuts >>>"
POWERSHELL_BLOCK_END = "# <<< pywayne toolsetup shortcuts <<<"


class SetupError(RuntimeError):
    """A user-facing setup error."""


def _print(message: str) -> None:
    print(message, flush=True)


def _success(message: str) -> None:
    _print("[OK] " + message)


def _warning(message: str) -> None:
    _print("[WARN] " + message)


def _finish_task(runner: "CommandRunner", message: str) -> None:
    if runner.dry_run:
        _print("[DRY-RUN] Plan validated: {} No changes made.".format(message))
    else:
        _success(message)


def _command_text(command: Sequence[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(list(command))
    return " ".join(shlex.quote(part) for part in command)


class CommandRunner:
    """Print commands before executing them, with a no-write dry-run mode."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run

    def run(
        self,
        command: Sequence[str],
        *,
        env: Optional[Dict[str, str]] = None,
        input_text: Optional[str] = None,
        check: bool = True,
    ) -> int:
        _print("$ " + _command_text(command))
        if self.dry_run:
            if input_text:
                for line in input_text.rstrip("\n").splitlines():
                    _print("  stdin> " + line)
            return 0

        process_env = os.environ.copy()
        if env:
            process_env.update(env)
        try:
            result = subprocess.run(
                list(command),
                env=process_env,
                input=input_text,
                text=input_text is not None,
                check=False,
            )
        except FileNotFoundError as exc:
            raise SetupError("Command not found: {}".format(command[0])) from exc

        if check and result.returncode != 0:
            raise SetupError(
                "Command failed with exit code {}: {}".format(
                    result.returncode, _command_text(command)
                )
            )
        return result.returncode


def normalize_platform(value: str) -> str:
    """Normalize CLI and Python platform names."""
    raw = platform_module.system() if value == "auto" else value
    normalized = raw.strip().lower()
    aliases = {
        "darwin": "macos",
        "mac": "macos",
        "macos": "macos",
        "linux": "linux",
        "win32": "windows",
        "win": "windows",
        "windows": "windows",
    }
    if normalized not in aliases:
        raise SetupError("Unsupported platform: {}".format(raw))
    return aliases[normalized]


def _shell_name(requested: str, target_platform: str) -> str:
    if requested != "auto":
        shell = requested.lower()
    elif target_platform == "windows":
        shell = "powershell"
    else:
        shell = Path(os.environ.get("SHELL", "")).name.lower()
        if shell not in {"bash", "zsh"}:
            shell = "zsh" if target_platform == "macos" else "bash"

    allowed = {"powershell"} if target_platform == "windows" else {"bash", "zsh"}
    if shell not in allowed:
        raise SetupError(
            "Shell '{}' is not supported on {}; choose one of: {}".format(
                shell, target_platform, ", ".join(sorted(allowed))
            )
        )
    return shell


def _powershell_profile_path() -> Path:
    for executable in ("pwsh", "powershell"):
        if not shutil.which(executable):
            continue
        try:
            output = subprocess.check_output(
                [executable, "-NoProfile", "-Command", "$PROFILE.CurrentUserCurrentHost"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            continue
        if output:
            return Path(output).expanduser()
    return Path.home() / "Documents" / "PowerShell" / "Microsoft.PowerShell_profile.ps1"


def resolve_profile_path(
    target_platform: str, shell: str, explicit_path: Optional[str]
) -> Path:
    if explicit_path:
        return Path(explicit_path).expanduser()
    if target_platform == "windows":
        return _powershell_profile_path()
    return Path.home() / (".zshrc" if shell == "zsh" else ".bashrc")


def _shell_quote(value: str) -> str:
    return shlex.quote(value)


UNIX_SHORTCUTS_TEMPLATE = r'''__BLOCK_START__
# Managed by toolsetup. Override these variables before calling proxy_on if needed.
PYWAYNE_HTTP_PROXY=${PYWAYNE_HTTP_PROXY:-__HTTP_PROXY__}
PYWAYNE_SOCKS_PROXY=${PYWAYNE_SOCKS_PROXY:-__SOCKS_PROXY__}
PYWAYNE_NO_PROXY=${PYWAYNE_NO_PROXY:-__NO_PROXY__}
PYWAYNE_GOTO_DB=${PYWAYNE_GOTO_DB:-"$HOME/.goto_paths"}
export PYWAYNE_HTTP_PROXY PYWAYNE_SOCKS_PROXY PYWAYNE_NO_PROXY PYWAYNE_GOTO_DB

proxy_off() {
  unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY no_proxy NO_PROXY
  if command -v git >/dev/null 2>&1; then
    git config --global --unset http.proxy >/dev/null 2>&1 || true
    git config --global --unset https.proxy >/dev/null 2>&1 || true
  fi
  printf '\033[31m%s\033[0m\n' "Proxy is OFF"
}

proxy_on() {
  proxy_off >/dev/null 2>&1

  export http_proxy="$PYWAYNE_HTTP_PROXY"
  export https_proxy="$PYWAYNE_HTTP_PROXY"
  export HTTP_PROXY="$PYWAYNE_HTTP_PROXY"
  export HTTPS_PROXY="$PYWAYNE_HTTP_PROXY"
  export all_proxy="$PYWAYNE_SOCKS_PROXY"
  export ALL_PROXY="$PYWAYNE_SOCKS_PROXY"
  export no_proxy="$PYWAYNE_NO_PROXY"
  export NO_PROXY="$PYWAYNE_NO_PROXY"

  if command -v git >/dev/null 2>&1; then
    git config --global http.proxy "$PYWAYNE_HTTP_PROXY"
    git config --global https.proxy "$PYWAYNE_HTTP_PROXY"
  fi

  printf '\033[32m%s\033[0m\n' "Proxy is ON"
  printf '  HTTP/HTTPS: %s\n' "$PYWAYNE_HTTP_PROXY"
  printf '  SOCKS5:     %s\n' "$PYWAYNE_SOCKS_PROXY"
  printf '  NO_PROXY:   %s\n' "$PYWAYNE_NO_PROXY"
}

_pywayne_goto_validate_key() {
  case "$1" in
    ""|*[!A-Za-z0-9_.-]*)
      echo "Shortcut must contain only letters, numbers, dot, underscore, or hyphen."
      return 1
      ;;
  esac
}

_pywayne_goto_absdir() {
  local path="$1"
  [ -n "$path" ] || return 1
  case "$path" in
    "~") path="$HOME" ;;
    "~/"*) path="$HOME/${path#\~/}" ;;
  esac
  (cd "$path" 2>/dev/null && pwd -P)
}

_pywayne_goto_list() {
  echo "Available shortcuts:"
  [ -f "$PYWAYNE_GOTO_DB" ] || return 0
  awk -F '\t' 'NF >= 2 {printf "  %-12s -> %s\n", $1, $2}' "$PYWAYNE_GOTO_DB" | sort
}

_pywayne_goto_set() {
  local key="$1" path="$2" tmp
  mkdir -p "$(dirname "$PYWAYNE_GOTO_DB")" || return 1
  [ -f "$PYWAYNE_GOTO_DB" ] || : >"$PYWAYNE_GOTO_DB"
  tmp="$(mktemp "${PYWAYNE_GOTO_DB}.tmp.XXXXXX")" || return 1
  if ! awk -F '\t' -v key="$key" '$1 != key {print}' "$PYWAYNE_GOTO_DB" >"$tmp"; then
    rm -f "$tmp"
    return 1
  fi
  printf '%s\t%s\n' "$key" "$path" >>"$tmp"
  mv "$tmp" "$PYWAYNE_GOTO_DB"
}

_pywayne_goto_del() {
  local key="$1" tmp
  [ -f "$PYWAYNE_GOTO_DB" ] || return 0
  tmp="$(mktemp "${PYWAYNE_GOTO_DB}.tmp.XXXXXX")" || return 1
  if ! awk -F '\t' -v key="$key" '$1 != key {print}' "$PYWAYNE_GOTO_DB" >"$tmp"; then
    rm -f "$tmp"
    return 1
  fi
  mv "$tmp" "$PYWAYNE_GOTO_DB"
}

goto() {
  local action="${1:-list}" key path target
  case "$action" in
    ls|list)
      _pywayne_goto_list
      ;;
    add)
      if [ "$#" -lt 3 ]; then
        echo "Usage: goto add <shortcut> <path>"
        return 1
      fi
      key="$2"
      shift 2
      path="$*"
      _pywayne_goto_validate_key "$key" || return 1
      if printf '%s' "$path" | LC_ALL=C grep '[[:cntrl:]]' >/dev/null 2>&1; then
        echo "Path cannot contain control characters."
        return 1
      fi
      path="$(_pywayne_goto_absdir "$path")" || { echo "Invalid directory: $path"; return 1; }
      _pywayne_goto_set "$key" "$path" || return 1
      echo "Added: $key -> $path"
      ;;
    rm|remove|del)
      key="$2"
      _pywayne_goto_validate_key "$key" || return 1
      if ! awk -F '\t' -v key="$key" '$1 == key {found=1} END {exit !found}' "$PYWAYNE_GOTO_DB" 2>/dev/null; then
        echo "Unknown shortcut: $key"
        return 1
      fi
      _pywayne_goto_del "$key" || return 1
      echo "Removed: $key"
      ;;
    clear|removeAll|rmAll)
      mkdir -p "$(dirname "$PYWAYNE_GOTO_DB")" || return 1
      : >"$PYWAYNE_GOTO_DB"
      echo "All shortcuts cleared."
      ;;
    *)
      target=""
      if [ -f "$PYWAYNE_GOTO_DB" ]; then
        target="$(awk -F '\t' -v key="$action" '$1 == key {print $2; exit}' "$PYWAYNE_GOTO_DB")"
      fi
      if [ -n "$target" ]; then
        cd "$target" || return 1
      else
        echo "Unknown shortcut: $action"
        echo "Tip: goto list"
        return 1
      fi
      ;;
  esac
}

if [ "$(uname -s 2>/dev/null)" = "Linux" ]; then
  gpu() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
      echo "nvidia-smi not found (install the NVIDIA driver and utilities first)."
      return 1
    fi

    local green yellow red cyan blue reset bold gpu_id pid mem info user elapsed cmd_full cmd_short script_name
    local any_proc=0
    if [ -t 1 ]; then
      green='\033[32m'; yellow='\033[33m'; red='\033[31m'; cyan='\033[36m'
      blue='\033[34m'; reset='\033[0m'; bold='\033[1m'
    else
      green=''; yellow=''; red=''; cyan=''; blue=''; reset=''; bold=''
    fi

    printf '%b\n' "${blue}${bold}======== GPU hardware overview ========${reset}"
    printf '%b%-3s %-25s %-6s %-6s %-8s %-20s %-6s%b\n' "$bold" \
      "ID" "Name" "Temp" "Fan" "Power" "VRAM" "Util" "$reset"
    nvidia-smi --query-gpu=index,name,temperature.gpu,fan.speed,power.draw,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader,nounits 2>/dev/null | \
      awk -F ', ' -v green="$green" -v yellow="$yellow" -v red="$red" -v reset="$reset" '
        {
          pct = ($7 + 0 > 0) ? (($6 + 0) / ($7 + 0)) * 100 : 0
          color = (pct > 80) ? red : ((pct > 50) ? yellow : green)
          name = substr($2, 1, 24)
          printf "%-3s %-25s %-6s %-6s %-8s %s%-6s / %-5s MB%s %-6s\n", \
            $1, name, $3 "C", $4 "%", $5 "W", color, $6, $7, reset, $8 "%"
        }'

    printf '\n%b\n' "${blue}${bold}======== GPU compute processes ========${reset}"
    printf '%b%-4s %-8s %-12s %-10s %-10s %-30s%b\n' "$bold" \
      "GPU" "PID" "User" "VRAM" "Elapsed" "Command/script" "$reset"

    while IFS= read -r gpu_id; do
      gpu_id="$(printf '%s' "$gpu_id" | tr -d '[:space:]')"
      [ -n "$gpu_id" ] || continue
      while IFS=',' read -r pid mem; do
        pid="$(printf '%s' "$pid" | tr -d '[:space:]')"
        mem="$(printf '%s' "$mem" | tr -cd '0-9')"
        [ -n "$pid" ] && [ "$pid" != "N/A" ] || continue
        any_proc=1
        info="$(ps -p "$pid" -o user= -o etime= -o args= 2>/dev/null)"
        if [ -n "$info" ]; then
          user="$(printf '%s\n' "$info" | awk '{print $1}')"
          elapsed="$(printf '%s\n' "$info" | awk '{print $2}')"
          cmd_full="$(printf '%s\n' "$info" | sed -E 's/^[[:space:]]*[^[:space:]]+[[:space:]]+[^[:space:]]+[[:space:]]*//')"
          if printf '%s' "$cmd_full" | grep -qi python; then
            script_name="$(printf '%s\n' "$cmd_full" | grep -oE '[^[:space:]]+\.py' | head -n 1)"
            if [ -n "$script_name" ]; then
              cmd_short="python $script_name"
            else
              cmd_short="${cmd_full:0:30}"
            fi
          else
            cmd_short="${cmd_full:0:30}"
          fi
        else
          user="Unknown"; elapsed="--"; cmd_short="--"
        fi
        printf '%-4s %-8s %b%-12s%b %b%-10s%b %-10s %-30s\n' \
          "$gpu_id" "$pid" "$cyan" "${user:0:11}" "$reset" \
          "$yellow" "${mem:-0}MB" "$reset" "$elapsed" "$cmd_short"
      done < <(nvidia-smi -i "$gpu_id" --query-compute-apps=pid,used_memory \
        --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d')
    done < <(nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null)

    if [ "$any_proc" -eq 0 ]; then
      printf '%b\n' "${yellow}(No visible compute processes. Graphics/driver allocations are not listed.)${reset}"
    fi

    printf '\n%b\n' "${blue}${bold}======== VRAM usage by user ========${reset}"
    printf '%b%-15s %-15s%b\n' "$bold" "User" "Total VRAM" "$reset"
    local aggregate
    aggregate="$(
      nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null | \
        while IFS=',' read -r pid mem; do
          pid="$(printf '%s' "$pid" | tr -d '[:space:]')"
          mem="$(printf '%s' "$mem" | tr -cd '0-9')"
          [ -n "$pid" ] && [ "$pid" != "N/A" ] || continue
          user="$(ps -o user= -p "$pid" 2>/dev/null | awk '{$1=$1; print}')"
          [ -n "$user" ] && printf '%s %s\n' "$user" "${mem:-0}"
        done | awk '{total[$1]+=$2} END {for (user in total) print user, total[user]}' | sort -k2,2nr
    )"
    if [ -z "$aggregate" ]; then
      printf '%b\n' "${yellow}(No visible compute processes.)${reset}"
    else
      printf '%s\n' "$aggregate" | awk -v cyan="$cyan" -v yellow="$yellow" -v reset="$reset" '
        {printf "%s%-15s%s %s%d MB%s\n", cyan, $1, reset, yellow, $2, reset}'
    fi
    printf '%s\n' "===================================="
  }
fi
__BLOCK_END__'''


POWERSHELL_SHORTCUTS_TEMPLATE = r'''__BLOCK_START__
# Managed by toolsetup. Override these variables before calling proxy_on if needed.
if (-not $env:PYWAYNE_HTTP_PROXY) { $env:PYWAYNE_HTTP_PROXY = __HTTP_PROXY__ }
if (-not $env:PYWAYNE_SOCKS_PROXY) { $env:PYWAYNE_SOCKS_PROXY = __SOCKS_PROXY__ }
if (-not $env:PYWAYNE_NO_PROXY) { $env:PYWAYNE_NO_PROXY = __NO_PROXY__ }
$script:PywayneGotoDb = Join-Path $HOME '.goto_paths.json'

function proxy_off {
  param([switch]$Quiet)
  Remove-Item Env:http_proxy,Env:https_proxy,Env:all_proxy,Env:no_proxy -ErrorAction SilentlyContinue
  Remove-Item Env:HTTP_PROXY,Env:HTTPS_PROXY,Env:ALL_PROXY,Env:NO_PROXY -ErrorAction SilentlyContinue
  if (Get-Command git -ErrorAction SilentlyContinue) {
    git config --global --unset http.proxy 2>$null | Out-Null
    git config --global --unset https.proxy 2>$null | Out-Null
  }
  if (-not $Quiet) { Write-Host 'Proxy is OFF' -ForegroundColor Red }
}

function proxy_on {
  proxy_off -Quiet
  $env:http_proxy = $env:PYWAYNE_HTTP_PROXY
  $env:https_proxy = $env:PYWAYNE_HTTP_PROXY
  $env:HTTP_PROXY = $env:PYWAYNE_HTTP_PROXY
  $env:HTTPS_PROXY = $env:PYWAYNE_HTTP_PROXY
  $env:all_proxy = $env:PYWAYNE_SOCKS_PROXY
  $env:ALL_PROXY = $env:PYWAYNE_SOCKS_PROXY
  $env:no_proxy = $env:PYWAYNE_NO_PROXY
  $env:NO_PROXY = $env:PYWAYNE_NO_PROXY
  if (Get-Command git -ErrorAction SilentlyContinue) {
    git config --global http.proxy $env:PYWAYNE_HTTP_PROXY | Out-Null
    git config --global https.proxy $env:PYWAYNE_HTTP_PROXY | Out-Null
  }
  Write-Host 'Proxy is ON' -ForegroundColor Green
  Write-Host "  HTTP/HTTPS: $env:PYWAYNE_HTTP_PROXY"
  Write-Host "  SOCKS5:     $env:PYWAYNE_SOCKS_PROXY"
  Write-Host "  NO_PROXY:   $env:PYWAYNE_NO_PROXY"
}

function _pywayne_goto_load {
  $table = @{}
  if (-not (Test-Path -LiteralPath $script:PywayneGotoDb)) { return $table }
  try {
    $raw = Get-Content -Raw -Encoding UTF8 -LiteralPath $script:PywayneGotoDb
    if (-not $raw.Trim()) { return $table }
    $object = $raw | ConvertFrom-Json
    foreach ($property in $object.PSObject.Properties) {
      $table[$property.Name] = [string]$property.Value
    }
    return $table
  } catch {
    throw "Cannot parse goto database: $script:PywayneGotoDb"
  }
}

function _pywayne_goto_save([hashtable]$Table) {
  $parent = Split-Path -Parent $script:PywayneGotoDb
  if (-not (Test-Path -LiteralPath $parent)) {
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
  }
  $Table | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 -LiteralPath $script:PywayneGotoDb
}

function goto {
  param(
    [Parameter(Position = 0)][string]$Action = 'list',
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest
  )
  $table = _pywayne_goto_load
  switch ($Action) {
    { $_ -in @('', 'ls', 'list') } {
      Write-Host 'Available shortcuts:'
      foreach ($key in ($table.Keys | Sort-Object)) {
        Write-Host ('  {0,-12} -> {1}' -f $key, $table[$key])
      }
      return
    }
    'add' {
      if ($Rest.Count -lt 2) { throw 'Usage: goto add <shortcut> <path>' }
      $key = $Rest[0]
      if ($key -notmatch '^[A-Za-z0-9_.-]+$') { throw 'Invalid shortcut name.' }
      $path = ($Rest | Select-Object -Skip 1) -join ' '
      $resolved = (Resolve-Path -LiteralPath $path -ErrorAction Stop).Path
      $table[$key] = $resolved
      _pywayne_goto_save $table
      Write-Host "Added: $key -> $resolved"
      return
    }
    { $_ -in @('rm', 'remove', 'del') } {
      if ($Rest.Count -ne 1) { throw 'Usage: goto remove <shortcut>' }
      $key = $Rest[0]
      if (-not $table.ContainsKey($key)) { throw "Unknown shortcut: $key" }
      $table.Remove($key) | Out-Null
      _pywayne_goto_save $table
      Write-Host "Removed: $key"
      return
    }
    { $_ -in @('clear', 'removeAll', 'rmAll') } {
      _pywayne_goto_save @{}
      Write-Host 'All shortcuts cleared.'
      return
    }
    default {
      if (-not $table.ContainsKey($Action)) {
        Write-Host "Unknown shortcut: $Action"
        Write-Host 'Tip: goto list'
        return 1
      }
      Set-Location -LiteralPath $table[$Action]
    }
  }
}
__BLOCK_END__'''


def build_unix_shortcuts_block(
    http_proxy: str, socks_proxy: str, no_proxy: str
) -> str:
    replacements = {
        "__BLOCK_START__": UNIX_BLOCK_START,
        "__BLOCK_END__": UNIX_BLOCK_END,
        "__HTTP_PROXY__": _shell_quote(http_proxy),
        "__SOCKS_PROXY__": _shell_quote(socks_proxy),
        "__NO_PROXY__": _shell_quote(no_proxy),
    }
    block = UNIX_SHORTCUTS_TEMPLATE
    for old, new in replacements.items():
        block = block.replace(old, new)
    return block.rstrip() + "\n"


def _powershell_quote(value: str) -> str:
    return "'{}'".format(value.replace("'", "''"))


def build_powershell_shortcuts_block(
    http_proxy: str, socks_proxy: str, no_proxy: str
) -> str:
    replacements = {
        "__BLOCK_START__": POWERSHELL_BLOCK_START,
        "__BLOCK_END__": POWERSHELL_BLOCK_END,
        "__HTTP_PROXY__": _powershell_quote(http_proxy),
        "__SOCKS_PROXY__": _powershell_quote(socks_proxy),
        "__NO_PROXY__": _powershell_quote(no_proxy),
    }
    block = POWERSHELL_SHORTCUTS_TEMPLATE
    for old, new in replacements.items():
        block = block.replace(old, new)
    return block.rstrip() + "\n"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode if path.exists() else None
    handle = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=str(path.parent), delete=False
    )
    temporary_path = Path(handle.name)
    try:
        with handle:
            handle.write(text)
        if mode is not None:
            os.chmod(str(temporary_path), mode)
        os.replace(str(temporary_path), str(path))
    except Exception:
        try:
            temporary_path.unlink()
        except OSError:
            pass
        raise


def ensure_managed_block(
    path: Path,
    block: str,
    start_marker: str,
    end_marker: str,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> str:
    """Append a marked block once, or replace it when force is requested."""
    original = path.read_text(encoding="utf-8") if path.exists() else ""
    start = original.find(start_marker)
    end = original.find(end_marker, start + len(start_marker)) if start >= 0 else -1

    if start >= 0 and end < 0:
        raise SetupError(
            "Found '{}' in {} without its closing marker; repair it manually first.".format(
                start_marker, path
            )
        )
    if start >= 0 and not force:
        _success("Shortcuts already exist in {}; skipped.".format(path))
        return "skipped"

    if start >= 0:
        suffix_start = end + len(end_marker)
        if suffix_start < len(original) and original[suffix_start] == "\n":
            suffix_start += 1
        updated = original[:start] + block + original[suffix_start:]
        action = "replace"
    else:
        separator = "" if not original else ("\n" if original.endswith("\n") else "\n\n")
        updated = original + separator + block
        action = "append"

    if dry_run:
        _print("[DRY-RUN] Would {} shortcuts block in {}".format(action, path))
        return action

    if path.exists() and original:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        backup = path.with_name(path.name + ".bak." + timestamp)
        shutil.copy2(str(path), str(backup))
        _print("Backup: {}".format(backup))
    _atomic_write(path, updated)
    action_text = "appended" if action == "append" else "replaced"
    _success("Shortcuts block {} in {}".format(action_text, path))
    return action


def _parse_goto_entries(values: Iterable[str]) -> List[Tuple[str, Path]]:
    entries: List[Tuple[str, Path]] = []
    for value in values:
        if "=" not in value:
            raise SetupError("Invalid --goto value '{}'; use KEY=PATH.".format(value))
        key, raw_path = value.split("=", 1)
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", key):
            raise SetupError("Invalid goto shortcut name: {}".format(key))
        if "\t" in raw_path or "\n" in raw_path:
            raise SetupError("Goto paths cannot contain tabs or newlines.")
        path = Path(raw_path).expanduser()
        if not path.is_dir():
            raise SetupError("Goto path is not an existing directory: {}".format(path))
        entries.append((key, path.resolve()))
    return entries


def _seed_unix_goto(entries: List[Tuple[str, Path]], dry_run: bool) -> None:
    if not entries:
        return
    db_path = Path.home() / ".goto_paths"
    if dry_run:
        for key, path in entries:
            _print("[DRY-RUN] Would add goto shortcut: {} -> {}".format(key, path))
        return
    existing: Dict[str, str] = {}
    if db_path.exists():
        for line in db_path.read_text(encoding="utf-8").splitlines():
            if "\t" in line:
                key, path = line.split("\t", 1)
                existing[key] = path
    for key, path in entries:
        existing[key] = str(path)
    text = "".join("{}\t{}\n".format(key, existing[key]) for key in sorted(existing))
    _atomic_write(db_path, text)
    _success("Updated {} goto shortcut(s) in {}".format(len(entries), db_path))


def _seed_windows_goto(entries: List[Tuple[str, Path]], dry_run: bool) -> None:
    if not entries:
        return
    db_path = Path.home() / ".goto_paths.json"
    if dry_run:
        for key, path in entries:
            _print("[DRY-RUN] Would add goto shortcut: {} -> {}".format(key, path))
        return
    existing: Dict[str, str] = {}
    if db_path.exists() and db_path.read_text(encoding="utf-8").strip():
        try:
            loaded = json.loads(db_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SetupError("Cannot parse {}: {}".format(db_path, exc)) from exc
        if not isinstance(loaded, dict):
            raise SetupError("Goto database must contain a JSON object: {}".format(db_path))
        existing.update({str(key): str(value) for key, value in loaded.items()})
    for key, path in entries:
        existing[key] = str(path)
    _atomic_write(db_path, json.dumps(existing, ensure_ascii=False, indent=2) + "\n")
    _success("Updated {} goto shortcut(s) in {}".format(len(entries), db_path))


def configure_shortcuts(args: argparse.Namespace, target_platform: str) -> None:
    shell = _shell_name(args.shell, target_platform)
    profile = resolve_profile_path(target_platform, shell, args.rc_file)
    if target_platform == "windows":
        block = build_powershell_shortcuts_block(
            args.http_proxy, args.socks_proxy, args.no_proxy
        )
        start_marker, end_marker = POWERSHELL_BLOCK_START, POWERSHELL_BLOCK_END
    else:
        block = build_unix_shortcuts_block(
            args.http_proxy, args.socks_proxy, args.no_proxy
        )
        start_marker, end_marker = UNIX_BLOCK_START, UNIX_BLOCK_END

    ensure_managed_block(
        profile,
        block,
        start_marker,
        end_marker,
        force=args.force,
        dry_run=args.dry_run,
    )
    entries = _parse_goto_entries(args.goto)
    if target_platform == "windows":
        _seed_windows_goto(entries, args.dry_run)
        _print("Reload with: . $PROFILE")
    else:
        _seed_unix_goto(entries, args.dry_run)
        _print("Reload with: source {}".format(shlex.quote(str(profile))))


def _sudo_prefix() -> List[str]:
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        return []
    return ["sudo"]


def _require_command(name: str, message: Optional[str] = None) -> None:
    if not shutil.which(name):
        raise SetupError(message or "Required command not found: {}".format(name))


def _linux_package_manager(dry_run: bool = False) -> str:
    for candidate in ("apt-get", "dnf", "yum", "zypper", "apk"):
        if shutil.which(candidate):
            return candidate
    if dry_run:
        _warning("No Linux package manager exists on this host; assuming apt-get for preview.")
        return "apt-get"
    raise SetupError("No supported Linux package manager found (apt, dnf, yum, zypper, apk).")


def _install_linux_node_prerequisites(runner: CommandRunner) -> None:
    manager = _linux_package_manager(runner.dry_run)
    sudo = _sudo_prefix()
    if manager == "apt-get":
        runner.run(sudo + [manager, "update"])
        runner.run(
            sudo
            + [manager, "install", "-y", "curl", "git", "build-essential", "ca-certificates"]
        )
    elif manager in {"dnf", "yum"}:
        runner.run(
            sudo
            + [manager, "install", "-y", "curl", "git", "gcc", "gcc-c++", "make", "ca-certificates"]
        )
    elif manager == "zypper":
        runner.run(
            sudo
            + [manager, "--non-interactive", "install", "curl", "git", "gcc", "gcc-c++", "make", "ca-certificates"]
        )
    else:
        runner.run(sudo + [manager, "add", "curl", "git", "bash", "build-base", "ca-certificates"])


def _validate_url(
    value: str, option_name: str, schemes: Sequence[str] = ("http", "https")
) -> None:
    parsed = urlparse(value)
    if parsed.scheme not in set(schemes) or not parsed.netloc:
        raise SetupError(
            "{} must use one of {}: {}".format(option_name, ", ".join(schemes), value)
        )


def _validate_versions(args: argparse.Namespace) -> None:
    if not re.fullmatch(r"v\d+\.\d+\.\d+", args.nvm_version):
        raise SetupError("--nvm-version must look like v0.40.6")
    if not re.fullmatch(r"[A-Za-z0-9.*_/-]+", args.node_version):
        raise SetupError("Invalid --node-version value")
    if not re.fullmatch(r"\d+\.\d+", args.mongodb_version):
        raise SetupError("--mongodb-version must look like 8.0")
    if args.mongodb_version != MONGODB_VERSION:
        raise SetupError(
            "This release has verified MongoDB {} only; got {}.".format(
                MONGODB_VERSION, args.mongodb_version
            )
        )
    _validate_url(args.npm_registry, "--npm-registry")
    _validate_url(args.http_proxy, "--http-proxy")
    _validate_url(
        args.socks_proxy,
        "--socks-proxy",
        schemes=("socks4", "socks4a", "socks5", "socks5h"),
    )


def _run_remote_script(
    runner: CommandRunner,
    url: str,
    shell: str,
    *,
    env: Optional[Dict[str, str]] = None,
) -> None:
    if runner.dry_run:
        runner.run(["curl", "-fsSL", url, "-o", "<temporary-install-script>"])
        runner.run([shell, "<temporary-install-script>"], env=env)
        return
    with tempfile.TemporaryDirectory(prefix="toolsetup-") as temp_dir:
        script_path = str(Path(temp_dir) / "install.sh")
        runner.run(["curl", "-fsSL", url, "-o", script_path])
        runner.run([shell, script_path], env=env)


def install_npm(args: argparse.Namespace, target_platform: str, runner: CommandRunner) -> None:
    if target_platform == "linux":
        _install_linux_node_prerequisites(runner)
    else:
        if not runner.dry_run and not shutil.which("curl"):
            raise SetupError("curl is required. Install Xcode Command Line Tools first.")
        if not runner.dry_run and not shutil.which("git"):
            raise SetupError("git is required. Run 'xcode-select --install' first.")
        if not runner.dry_run:
            if not shutil.which("xcode-select") or runner.run(
                ["xcode-select", "-p"], check=False
            ) != 0:
                raise SetupError(
                    "Xcode Command Line Tools are required. Run 'xcode-select --install', "
                    "wait for it to finish, then rerun toolsetup."
                )

    shell = _shell_name(args.shell, target_platform)
    profile = resolve_profile_path(target_platform, shell, args.rc_file)
    if not runner.dry_run:
        profile.parent.mkdir(parents=True, exist_ok=True)
        profile.touch(exist_ok=True)

    nvm_dir = Path(os.environ.get("NVM_DIR", str(Path.home() / ".nvm"))).expanduser()
    nvm_script = nvm_dir / "nvm.sh"
    if nvm_script.exists() and not args.force:
        _success("nvm already exists at {}; installer skipped.".format(nvm_script))
    else:
        install_url = "https://raw.githubusercontent.com/nvm-sh/nvm/{}/install.sh".format(
            args.nvm_version
        )
        _run_remote_script(
            runner,
            install_url,
            "bash",
            env={"PROFILE": str(profile), "NVM_DIR": str(nvm_dir)},
        )

    nvm_commands = r'''
set -e
export NVM_DIR="$TOOLSETUP_NVM_DIR"
[ -s "$NVM_DIR/nvm.sh" ] || { echo "nvm.sh not found: $NVM_DIR/nvm.sh" >&2; exit 1; }
. "$NVM_DIR/nvm.sh"
nvm --version
nvm ls-remote --lts
nvm install "$TOOLSETUP_NODE_VERSION"
nvm alias default "$TOOLSETUP_NODE_VERSION"
nvm use default
node -v
npm -v
npm config set registry "$TOOLSETUP_NPM_REGISTRY"
npm config get registry
'''.strip()
    runner.run(
        ["bash", "-c", nvm_commands],
        env={
            "TOOLSETUP_NVM_DIR": str(nvm_dir),
            "TOOLSETUP_NODE_VERSION": args.node_version,
            "TOOLSETUP_NPM_REGISTRY": args.npm_registry,
        },
    )
    _finish_task(
        runner,
        "Node.js/npm configured; open a new shell or source {}.".format(profile),
    )


def install_tailscale(runner: CommandRunner) -> None:
    _run_remote_script(runner, "https://tailscale.com/install.sh", "sh")
    sudo = _sudo_prefix()
    if runner.dry_run or (shutil.which("systemctl") and Path("/run/systemd/system").exists()):
        runner.run(sudo + ["systemctl", "enable", "--now", "tailscaled"])
    else:
        _warning("systemd is not active; relying on the distribution installer to start tailscaled.")
    runner.run(sudo + ["tailscale", "up"])
    _finish_task(runner, "Tailscale installed and started.")


def _read_os_release() -> Dict[str, str]:
    path = Path("/etc/os-release")
    if not path.exists():
        return {}
    values: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw_line or raw_line.startswith("#") or "=" not in raw_line:
            continue
        key, value = raw_line.split("=", 1)
        values[key] = value.strip().strip('"').strip("'")
    return values


def _install_mongodb_apt(
    args: argparse.Namespace, runner: CommandRunner, os_release: Dict[str, str]
) -> None:
    distro = os_release.get("ID", "").lower()
    codename = os_release.get("VERSION_CODENAME", "").lower()
    supported = {
        "ubuntu": {"focal", "jammy", "noble"},
        "debian": {"bookworm"},
    }
    if distro not in supported or codename not in supported[distro]:
        raise SetupError(
            "MongoDB {} packages support Ubuntu 20.04/22.04/24.04 or Debian 12; "
            "detected {} {}.".format(args.mongodb_version, distro or "unknown", codename or "unknown")
        )
    architecture = platform_module.machine().lower()
    if architecture not in {"x86_64", "amd64", "aarch64", "arm64"}:
        raise SetupError("MongoDB requires a supported 64-bit architecture; detected {}.".format(architecture))
    if distro == "debian" and architecture not in {"x86_64", "amd64"}:
        raise SetupError("MongoDB 8.0 Community packages for Debian 12 require x86_64.")

    if not runner.dry_run and shutil.which("dpkg-query"):
        package_status = subprocess.run(
            ["dpkg-query", "-W", "-f=${Status}", "mongodb"],
            text=True,
            capture_output=True,
            check=False,
        )
        if package_status.returncode == 0 and "install ok installed" in package_status.stdout:
            raise SetupError(
                "The distro-provided 'mongodb' package conflicts with mongodb-org. "
                "Remove it explicitly, review your data, then rerun toolsetup."
            )

    sudo = _sudo_prefix()
    runner.run(sudo + ["apt-get", "update"])
    runner.run(
        sudo
        + ["apt-get", "install", "-y", "gnupg", "curl", "ca-certificates", "python3-venv"]
    )

    keyring = "/usr/share/keyrings/mongodb-server-{}.gpg".format(args.mongodb_version)
    key_url = "https://pgp.mongodb.com/server-{}.asc".format(args.mongodb_version)
    if runner.dry_run:
        runner.run(["curl", "-fsSL", key_url, "-o", "<temporary-mongodb-key>"])
        runner.run(sudo + ["gpg", "--batch", "--yes", "--dearmor", "-o", keyring, "<temporary-mongodb-key>"])
    else:
        with tempfile.TemporaryDirectory(prefix="toolsetup-mongodb-") as temp_dir:
            key_path = str(Path(temp_dir) / "server.asc")
            runner.run(["curl", "-fsSL", key_url, "-o", key_path])
            runner.run(
                sudo + ["gpg", "--batch", "--yes", "--dearmor", "-o", keyring, key_path]
            )

    if distro == "ubuntu":
        repo = (
            "deb [ arch=amd64,arm64 signed-by={keyring} ] "
            "https://repo.mongodb.org/apt/ubuntu {codename}/mongodb-org/{version} multiverse\n"
        ).format(keyring=keyring, codename=codename, version=args.mongodb_version)
    else:
        repo = (
            "deb [ signed-by={keyring} ] https://repo.mongodb.org/apt/debian "
            "{codename}/mongodb-org/{version} main\n"
        ).format(keyring=keyring, codename=codename, version=args.mongodb_version)
    list_path = "/etc/apt/sources.list.d/mongodb-org-{}.list".format(args.mongodb_version)
    runner.run(sudo + ["tee", list_path], input_text=repo)
    runner.run(sudo + ["apt-get", "update"])
    runner.run(sudo + ["apt-get", "install", "-y", "mongodb-org"])
    runner.run(sudo + ["systemctl", "enable", "--now", "mongod"])


def _install_mongodb_rhel(
    args: argparse.Namespace, runner: CommandRunner, os_release: Dict[str, str]
) -> None:
    version_id = os_release.get("VERSION_ID", "").split(".", 1)[0]
    if version_id not in {"8", "9"}:
        raise SetupError("MongoDB {} supports RHEL-compatible Linux 8 or 9.".format(args.mongodb_version))
    machine = platform_module.machine().lower()
    architecture = "aarch64" if machine in {"aarch64", "arm64"} else "x86_64"
    repo = """[mongodb-org-{version}]
name=MongoDB Repository
baseurl=https://repo.mongodb.org/yum/redhat/{rhel}/mongodb-org/{version}/{arch}/
gpgcheck=1
enabled=1
gpgkey=https://pgp.mongodb.com/server-{version}.asc
""".format(version=args.mongodb_version, rhel=version_id, arch=architecture)
    repo_path = "/etc/yum.repos.d/mongodb-org-{}.repo".format(args.mongodb_version)
    sudo = _sudo_prefix()
    manager = "dnf" if shutil.which("dnf") else "yum"
    runner.run(sudo + ["tee", repo_path], input_text=repo)
    runner.run(sudo + [manager, "install", "-y", "mongodb-org", "python3"])
    runner.run(sudo + ["systemctl", "enable", "--now", "mongod"])


def _install_pymongo(args: argparse.Namespace, runner: CommandRunner) -> None:
    if args.skip_pymongo:
        _warning("PyMongo installation skipped by request.")
        return
    python_executable = str(Path(args.python).expanduser()) if args.python else sys.executable
    venv_dir = Path(args.pymongo_venv).expanduser()
    venv_python = venv_dir / "bin" / "python"
    if not venv_python.exists():
        runner.run([python_executable, "-m", "venv", str(venv_dir)])
    else:
        _success("PyMongo virtual environment already exists: {}".format(venv_dir))
    runner.run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "pymongo"])
    runner.run(
        [
            str(venv_python),
            "-c",
            "import pymongo; print('PyMongo', pymongo.version)",
        ]
    )
    if runner.dry_run:
        _print("[DRY-RUN] Planned PyMongo environment: {}".format(venv_dir))
    else:
        _success("PyMongo environment: {}".format(venv_dir))
    _print("Activate with: source {}".format(shlex.quote(str(venv_dir / "bin" / "activate"))))


def install_mongodb(
    args: argparse.Namespace, target_platform: str, runner: CommandRunner
) -> None:
    if target_platform == "macos":
        if not runner.dry_run:
            _require_command(
                "brew",
                "Homebrew is required. Install it from https://brew.sh, then rerun toolsetup.",
            )
        formula = "mongodb-community@{}".format(args.mongodb_version)
        runner.run(["brew", "tap", "mongodb/brew"])
        runner.run(["brew", "update"])
        runner.run(["brew", "install", formula])
        runner.run(["brew", "services", "start", formula])
    else:
        if not runner.dry_run:
            _require_command(
                "systemctl",
                "MongoDB automatic startup on Linux requires a systemd-based distribution.",
            )
        os_release = _read_os_release()
        if runner.dry_run and not os_release.get("ID"):
            _warning("No Linux os-release exists on this host; assuming Ubuntu 24.04 for preview.")
            os_release = {
                "ID": "ubuntu",
                "VERSION_ID": "24.04",
                "VERSION_CODENAME": "noble",
            }
        distro = os_release.get("ID", "").lower()
        distro_like = set(os_release.get("ID_LIKE", "").lower().split())
        if distro in {"ubuntu", "debian"}:
            _install_mongodb_apt(args, runner, os_release)
        elif distro in {"rhel", "centos", "rocky", "almalinux", "ol"} or "rhel" in distro_like:
            _install_mongodb_rhel(args, runner, os_release)
        else:
            raise SetupError(
                "MongoDB automation currently supports Ubuntu, Debian, RHEL, Rocky, AlmaLinux, "
                "CentOS Stream, Oracle Linux, and macOS; detected {}.".format(distro or "unknown Linux")
            )

    _install_pymongo(args, runner)
    _finish_task(
        runner,
        "MongoDB {} installed and configured to start automatically.".format(
            args.mongodb_version
        ),
    )


TASK_SUPPORT = {
    "shortcuts": {"macos", "linux", "windows"},
    "npm": {"macos", "linux"},
    "tailscale": {"linux"},
    "mongodb": {"macos", "linux"},
}


def selected_tasks(task: str, target_platform: str) -> List[str]:
    if task != "all":
        if target_platform not in TASK_SUPPORT[task]:
            raise SetupError("Task '{}' is not supported on {}.".format(task, target_platform))
        return [task]
    order = ["shortcuts", "npm", "tailscale", "mongodb"]
    return [name for name in order if target_platform in TASK_SUPPORT[name]]


def _confirm_install(tasks: List[str], target_platform: str, assume_yes: bool, dry_run: bool) -> None:
    installers = [task for task in tasks if task != "shortcuts"]
    if not installers or assume_yes or dry_run:
        return
    if not sys.stdin.isatty():
        raise SetupError("Installation requires confirmation; rerun with --yes or --dry-run.")
    prompt = "Run {} on {}? This may download software and use sudo. [y/N] ".format(
        ", ".join(installers), target_platform
    )
    if input(prompt).strip().lower() not in {"y", "yes"}:
        raise SetupError("Cancelled.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Configure shell shortcuts and install common developer tools on macOS, Linux, and Windows."
        )
    )
    parser.add_argument(
        "--task",
        choices=["shortcuts", "npm", "tailscale", "mongodb", "all"],
        required=True,
        help="Configuration task to run.",
    )
    parser.add_argument(
        "--platform",
        default="auto",
        choices=["auto", "macos", "mac", "linux", "windows", "win"],
        help="Target platform (default: detect current host).",
    )
    parser.add_argument(
        "--shell",
        default="auto",
        choices=["auto", "bash", "zsh", "powershell"],
        help="Shell profile to configure.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without changing the system.")
    parser.add_argument("--yes", action="store_true", help="Skip the installation confirmation prompt.")
    parser.add_argument("--force", action="store_true", help="Replace managed config or rerun an existing installer.")

    shortcuts = parser.add_argument_group("shortcuts")
    shortcuts.add_argument("--rc-file", help="Override the shell profile path.")
    shortcuts.add_argument("--http-proxy", default=HTTP_PROXY)
    shortcuts.add_argument("--socks-proxy", default=SOCKS_PROXY)
    shortcuts.add_argument("--no-proxy", default=NO_PROXY)
    shortcuts.add_argument(
        "--goto",
        action="append",
        default=[],
        metavar="KEY=PATH",
        help="Seed a persistent goto shortcut; repeat for multiple entries.",
    )

    npm = parser.add_argument_group("npm")
    npm.add_argument("--nvm-version", default=NVM_VERSION)
    npm.add_argument("--node-version", default=NODE_VERSION)
    npm.add_argument("--npm-registry", default=NPM_REGISTRY)

    mongodb = parser.add_argument_group("mongodb")
    mongodb.add_argument("--mongodb-version", default=MONGODB_VERSION)
    mongodb.add_argument(
        "--pymongo-venv",
        default=str(Path.home() / ".venvs" / "pywayne-mongodb"),
        help="Virtual environment used for PyMongo.",
    )
    mongodb.add_argument("--python", help="Python executable used to create the PyMongo virtual environment.")
    mongodb.add_argument("--skip-pymongo", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_versions(args)
        target_platform = normalize_platform(args.platform)
        host_platform = normalize_platform("auto")
        if target_platform != host_platform and not args.dry_run:
            raise SetupError(
                "Target platform {} does not match this {} host. Use --dry-run to inspect another platform."
                .format(target_platform, host_platform)
            )
        tasks = selected_tasks(args.task, target_platform)
        _print("Platform: {} | Tasks: {}".format(target_platform, ", ".join(tasks)))
        _confirm_install(tasks, target_platform, args.yes, args.dry_run)
        runner = CommandRunner(dry_run=args.dry_run)
        for task in tasks:
            _print("\n== {} ==".format(task))
            if task == "shortcuts":
                configure_shortcuts(args, target_platform)
            elif task == "npm":
                install_npm(args, target_platform, runner)
            elif task == "tailscale":
                install_tailscale(runner)
            elif task == "mongodb":
                install_mongodb(args, target_platform, runner)
        if args.dry_run:
            _success("Dry-run completed; no changes were made.")
        else:
            _success("All requested tasks completed.")
        return 0
    except (SetupError, OSError) as exc:
        print("{}: {}".format(TOOL_NAME, exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
