#!/usr/bin/env bash
# Every test that needs no GPU, in about two minutes on an 8-core CPU:
#   1. the capture plugin's bookkeeping, with stand-ins for vLLM's runner state;
#   2. the scoring network over stored rows against Cloudflare's, on the 72 test records;
#   3. the decision service against a stand-in for vLLM (service/mock_vllm.py) that writes capture
#      files with the plugin's own code and imitates vLLM's prefix cache: a game whose state grows,
#      sent to a service that reuses prompt starts and to one that reads every prompt in full, which
#      must agree; then, if saved hidden states are given, the 72 records against the reference answers.
#
# usage: eval/run-cpu-tests.sh RELEASE_DIR [HIDDEN_DIR]
#   RELEASE_DIR  the Clef release files (service/fetch_release.py)
#   HIDDEN_DIR   optional: the hidden states saved by eval/reference.py, published with the adapter
#   PYTHON       environment variable naming the interpreter (default python3)
set -euo pipefail
RELEASE=${1:?usage: run-cpu-tests.sh RELEASE_DIR [HIDDEN_DIR]}
HIDDEN=${2:-}
PYTHON=${PYTHON:-python3}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK=$(mktemp -d)
PIDS=()
cleanup() { for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done; rm -rf "$WORK"; }
trap cleanup EXIT
wait_up() { for _ in $(seq 120); do curl -sf -o /dev/null "$1/health" && return 0; sleep 0.5; done; echo "$1 did not start"; return 1; }

echo "=== 1. capture plugin"
$PYTHON -I "$ROOT/vllm-plugin/test_capture.py"

[ -f "$ROOT/eval/records.jsonl" ] || $PYTHON -I "$ROOT/eval/build_records.py"

echo "=== 2. scoring network over stored rows"
$PYTHON -I "$ROOT/service/test_reuse.py" --release "$RELEASE" ${HIDDEN:+--reference-hidden "$HIDDEN"}

echo "=== 3. decision service against a stand-in for vLLM"
$PYTHON -I "$ROOT/service/mock_vllm.py" "$WORK/capture" --listen 127.0.0.1:18199 --block 256 \
    ${HIDDEN:+--reference-hidden "$HIDDEN"} > "$WORK/mock.log" 2>&1 & PIDS+=($!)
$PYTHON -I "$ROOT/service/service.py" --release "$RELEASE" --vllm http://127.0.0.1:18199 \
    --capture-dir "$WORK/capture" --listen 127.0.0.1:18191 > "$WORK/reuse.log" 2>&1 & PIDS+=($!)
$PYTHON -I "$ROOT/service/service.py" --release "$RELEASE" --vllm http://127.0.0.1:18199 \
    --capture-dir "$WORK/capture" --listen 127.0.0.1:18192 --reuse-memory-mb 0 > "$WORK/full.log" 2>&1 & PIDS+=($!)
wait_up http://127.0.0.1:18199 && wait_up http://127.0.0.1:18191 && wait_up http://127.0.0.1:18192
$PYTHON -I "$ROOT/eval/game_test.py" http://127.0.0.1:18191 http://127.0.0.1:18192 --release "$RELEASE" \
    --corpus-tokens 4000 --decisions 6 --block 256 --held-back-blocks 0 --check-refusal
if [ -n "$HIDDEN" ]; then
    echo "--- the 72 records against the reference answers (the stand-in serves the saved hidden states)"
    $PYTHON -I "$ROOT/eval/test_service.py" http://127.0.0.1:18192
else
    echo "--- skipped: the 72 records against the reference answers (needs HIDDEN_DIR)"
fi

echo "=== 4. priming, against a stand-in for vLLM with a 16-token matching unit"
$PYTHON -I "$ROOT/service/mock_vllm.py" "$WORK/capture2" --listen 127.0.0.1:18299 --block 256 --match-unit 16 \
    > "$WORK/mock2.log" 2>&1 & PIDS+=($!)
$PYTHON -I "$ROOT/service/service.py" --release "$RELEASE" --vllm http://127.0.0.1:18299 \
    --capture-dir "$WORK/capture2" --listen 127.0.0.1:18291 --prime-margin 64 --prime-unit 16 > "$WORK/prime.log" 2>&1 & PIDS+=($!)
$PYTHON -I "$ROOT/service/service.py" --release "$RELEASE" --vllm http://127.0.0.1:18299 \
    --capture-dir "$WORK/capture2" --listen 127.0.0.1:18292 --reuse-memory-mb 0 > "$WORK/full2.log" 2>&1 & PIDS+=($!)
wait_up http://127.0.0.1:18299 && wait_up http://127.0.0.1:18291 && wait_up http://127.0.0.1:18292
$PYTHON -I "$ROOT/eval/game_test.py" http://127.0.0.1:18291 http://127.0.0.1:18292 --release "$RELEASE" \
    --corpus-tokens 4000 --decisions 6 --block 256 --match-unit 16 --prime-margin 64
echo "all CPU tests passed"
