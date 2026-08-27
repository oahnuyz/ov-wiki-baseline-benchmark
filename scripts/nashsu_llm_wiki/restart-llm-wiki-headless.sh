#!/usr/bin/env bash
set -Eeuo pipefail

deploy_root="${LLM_WIKI_DEPLOY_ROOT:-/home/ZhangYunhao/nashsu-llm-wiki-baseline}"
launcher="$deploy_root/tools/run-llm-wiki-headless.sh"
runtime_dir="$deploy_root/run"
pid_file="$runtime_dir/llm-wiki-headless.pid"
log_file="$deploy_root/logs/llm-wiki-headless.log"

[[ -x "$launcher" ]] || {
  printf 'Missing executable launcher: %s\n' "$launcher" >&2
  exit 2
}

mkdir -p "$runtime_dir" "$deploy_root/logs"

"$deploy_root/tools/stop-llm-wiki-headless.sh"

nohup setsid "$launcher" >>"$log_file" 2>&1 </dev/null &
new_pid=$!
printf '%s\n' "$new_pid" >"$pid_file"

sleep 0.5
if ! kill -0 "$new_pid" 2>/dev/null; then
  printf 'LLM Wiki service failed to start; inspect %s\n' "$log_file" >&2
  exit 1
fi

printf 'LLM Wiki headless service restarted: pid=%s\n' "$new_pid"
