#!/usr/bin/env bash
# Qwen3.8-27B in stock vLLM, serving chat as usual and Clef decisions through the adapter.
#
# Before running it:
#   pip install ./vllm-plugin          (into the environment vLLM runs in)
#   the adapter in $CLEF_ADAPTER_DIR   (downloaded from Hugging Face, or built by extract/extract_adapter.py)
# Then start the decision service against port 8000 (README, step 3).
#
# Untested in this form: the plugin has run on the GPU only in the DFlash2 image
# (examples/rtx5090-nvfp4-dflash2). Run eval/compare_hidden.py and eval/test_service.py on your
# setup and compare with the published results before relying on it.
set -euo pipefail
: "${CLEF_ADAPTER_DIR:?set CLEF_ADAPTER_DIR to the adapter directory}"
MODEL=${MODEL:-Qwen/Qwen3.8-27B}           # or a quantized Qwen3.8-27B checkpoint

# The plugin writes the hidden states of decision requests here; the decision service reads them.
export CLEF_CAPTURE_DIR=/dev/shm/clef-hidden
# Give adapter buffers only to the layers Clef changed (about 2.3 GB saved).
export CLEF_LORA_LAYERS=40-63
# With a speculative draft model: let decisions reuse every whole cached block (tested on the
# DFlash2 fork of vLLM 0.27.2 only; see docs/how-it-works.md#how-much-vllm-reuses).
# export CLEF_FULL_CACHE_HITS=1

exec vllm serve "$MODEL" \
    --served-model-name qwen3.8-27b \
    --enable-lora \
    --max-lora-rank 256 \
    --max-loras 1 \
    --max-cpu-loras 1 \
    --lora-modules clef="$CLEF_ADAPTER_DIR" \
    --enable-prefix-caching \
    "$@"
