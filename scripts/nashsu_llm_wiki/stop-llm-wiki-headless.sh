#!/usr/bin/env bash
set -Eeuo pipefail

deploy_root="${LLM_WIKI_DEPLOY_ROOT:-/home/ZhangYunhao/nashsu-llm-wiki-baseline}"
pid_file="$deploy_root/run/llm-wiki-headless.pid"

[[ -f "$pid_file" ]] || exit 0
old_pid="$(tr -dc '0-9' <"$pid_file")"
[[ -n "$old_pid" ]] || exit 0

if kill -0 "$old_pid" 2>/dev/null; then
  /bin/kill -TERM -- "-$old_pid" 2>/dev/null || /bin/kill -TERM "$old_pid" 2>/dev/null || true
  for _ in $(seq 1 50); do
    kill -0 "$old_pid" 2>/dev/null || break
    sleep 0.2
  done
  if kill -0 "$old_pid" 2>/dev/null; then
    /bin/kill -KILL -- "-$old_pid" 2>/dev/null || /bin/kill -KILL "$old_pid" 2>/dev/null
  fi
fi

rm -f -- "$pid_file"
