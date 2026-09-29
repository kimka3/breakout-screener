#!/usr/bin/env bash
set -Eeuo pipefail

if (( EUID != 0 )); then
  printf 'Run this installer with sudo.\n' >&2
  exit 1
fi

for command in curl systemctl install; do
  if ! command -v "$command" >/dev/null 2>&1; then
    printf 'Required command is missing: %s\n' "$command" >&2
    exit 1
  fi
done

source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
token="${GITHUB_TOKEN:-}"
if [[ -z "$token" ]]; then
  if [[ ! -t 0 ]]; then
    printf 'Set GITHUB_TOKEN or run interactively to enter it.\n' >&2
    exit 1
  fi
  read -r -s -p 'GitHub fine-grained token: ' token
  printf '\n'
fi
if [[ ! "$token" =~ ^[A-Za-z0-9_]+$ ]]; then
  printf 'The GitHub token contains unexpected characters.\n' >&2
  exit 1
fi

temp_env="$(mktemp)"
trap 'rm -f -- "$temp_env"' EXIT
printf 'GITHUB_TOKEN=%s\n' "$token" >"$temp_env"

install -d -m 0755 /usr/local/libexec/breakout-screener
install -m 0755 "$source_dir/trigger-breakout.sh" \
  /usr/local/libexec/breakout-screener/trigger-breakout.sh
install -m 0644 "$source_dir/breakout-trigger.service" \
  /etc/systemd/system/breakout-trigger.service
install -m 0644 "$source_dir/breakout-trigger.timer" \
  /etc/systemd/system/breakout-trigger.timer
install -m 0600 "$temp_env" /etc/breakout-screener.env

GITHUB_TOKEN="$token" \
  /usr/local/libexec/breakout-screener/trigger-breakout.sh --check

systemctl daemon-reload
systemctl enable --now breakout-trigger.timer

printf '\nInstalled successfully.\n'
systemctl --no-pager status breakout-trigger.timer || true
printf '\nNext run:\n'
systemctl list-timers breakout-trigger.timer --no-pager
printf '\nThe timer only dispatches between 07:00 and 08:49 KST.\n'
