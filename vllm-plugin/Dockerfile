# A vLLM image plus the clef_capture plugin. Requests that do not ask for capture are served
# exactly as by the base image.
#
#   docker build -t vllm-clef --build-arg BASE=vllm/vllm-openai:v0.31.0 vllm-plugin
#
# BASE is the image you already run. The plugin was run on the GPU in the DFlash2 image for RTX
# 50-series cards (examples/rtx5090-nvfp4-dflash2) and checked against the source of vLLM 0.31.0.
ARG BASE
FROM ${BASE}
USER root
COPY pyproject.toml /opt/clef-capture/pyproject.toml
COPY clef_capture /opt/clef-capture/clef_capture
RUN python3 -m pip install --no-deps --no-build-isolation --no-cache-dir /opt/clef-capture
