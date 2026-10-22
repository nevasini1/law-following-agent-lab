#!/usr/bin/env bash
# Build llama.cpp's llama-server (CPU, native instructions) into ./tools/llama.cpp.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p tools && cd tools
[ -d llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build --target llama-server -j "$(nproc)"
