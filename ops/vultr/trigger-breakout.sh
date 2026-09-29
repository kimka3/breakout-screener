#!/usr/bin/env bash
set -Eeuo pipefail

readonly REPOSITORY="kimka3/breakout-screener"
readonly WORKFLOW="daily-schedule.yml"
readonly REF="main"
readonly API_ROOT="https://api.github.com/repos/${REPOSITORY}"

log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*"
}

require_token() {
  if [[ -z "${GITHUB_TOKEN:-}" ]]; then
    log "GITHUB_TOKEN is not set"
    exit 2
  fi
  # Fine-grained PATs currently contain only letters, digits and underscores.
  if [[ ! "$GITHUB_TOKEN" =~ ^[A-Za-z0-9_]+$ ]]; then
    log "GITHUB_TOKEN contains unexpected characters"
    exit 2
  fi
}

request() {
  local method="$1"
  local url="$2"
  local data="${3:-}"
  local status
  local -a curl_args=(--config -)
  local null_device="/dev/null"

  # Git for Windows uses Schannel; this is ignored on the Linux VPS.
  if [[ "$(uname -s)" == MINGW* ]]; then
    curl_args+=(--ssl-no-revoke)
    null_device="NUL"
  fi

  if ! status="$({
    printf 'url = "%s"\n' "$url"
    printf 'request = "%s"\n' "$method"
    printf 'header = "Accept: application/vnd.github+json"\n'
    printf 'header = "Content-Type: application/json"\n'
    printf 'header = "X-GitHub-Api-Version: 2022-11-28"\n'
    printf 'header = "User-Agent: breakout-screener-vultr-trigger"\n'
    printf 'header = "Authorization: Bearer %s"\n' "$GITHUB_TOKEN"
    if [[ -n "$data" ]]; then
      printf 'data = "%s"\n' "$data"
    fi
    printf 'output = "%s"\n' "$null_device"
    printf 'silent\nshow-error\n'
    printf 'write-out = "%%{http_code}"\n'
  } | curl "${curl_args[@]}")"; then
    return 1
  fi
  printf '%s' "$status"
}

require_token

case "${1:-}" in
  --check)
    if ! status="$(request GET "${API_ROOT}/actions/workflows/${WORKFLOW}")"; then
      log "GitHub credential check failed because the API request could not be completed"
      exit 1
    fi
    if [[ "$status" != "200" ]]; then
      log "GitHub credential check failed (HTTP ${status})"
      exit 1
    fi
    log "GitHub credential and workflow check succeeded"
    exit 0
    ;;
  --force)
    ;;
  "")
    now_hhmm="$(TZ=Asia/Seoul date +%H%M)"
    now_number=$((10#$now_hhmm))
    # A delayed timer must not refresh Korean currentPrice after the 09:00 open.
    if (( now_number < 700 || now_number >= 850 )); then
      log "Skipped outside the 07:00-08:49 KST safety window (now ${now_hhmm} KST)"
      exit 0
    fi
    ;;
  *)
    printf 'usage: %s [--check|--force]\n' "$0" >&2
    exit 2
    ;;
esac

if ! status="$(request POST "${API_ROOT}/actions/workflows/${WORKFLOW}/dispatches" '{\"ref\":\"main\"}')"; then
  log "GitHub workflow dispatch failed because the API request could not be completed"
  exit 1
fi
if [[ "$status" != "204" ]]; then
  log "GitHub workflow dispatch failed (HTTP ${status})"
  exit 1
fi

log "Dispatched ${REPOSITORY}/${WORKFLOW} on ${REF}"
