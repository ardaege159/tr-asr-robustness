#!/usr/bin/env bash
# Keeps every model's plan going until the speech data is complete and a full pass neither adds
# rows nor stops on a quota. Each model runs in its own process because each has its own quota.
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
MODELS="gemini:gemini-3.1-flash-lite groq:openai/gpt-oss-120b groq:openai/gpt-oss-20b groq:qwen/qwen3.8-27b"
while true; do
  before=$(cat results/raw/*.jsonl 2>/dev/null | wc -l)
  rm -f pass_*.tmp
  for m in $MODELS; do
    name=$(echo "$m" | tr ':/' '__')
    .venv/Scripts/python -X utf8 -m trasr.run --plan --models "$m" > "pass_$name.tmp" 2>&1 &
  done
  wait
  quota=$(cat pass_*.tmp | grep -c "daily quota used up")
  for f in pass_*.tmp; do base=${f#pass_}; cat "$f" >> "run_${base%.tmp}.log"; done
  rm -f pass_*.tmp
  after=$(cat results/raw/*.jsonl 2>/dev/null | wc -l)
  echo "$(date '+%F %T') pass done: $before -> $after rows, $quota model(s) waiting on quota"
  if [ -f speech.done ] && [ "$after" -eq "$before" ] && [ "$quota" -eq 0 ]; then echo finished; break; fi
  sleep 600
done
