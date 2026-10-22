"""Open-weight model variants used across experiments. All are ungated downloads; no API keys anywhere."""
import os
from dataclasses import dataclass
from pathlib import Path

from lawful.paths import ROOT

# Override with LAWFUL_MODELS (gguf/ and hf/ subfolders) and LAWFUL_LLAMA_SERVER (path to llama.cpp's llama-server).
MODEL_DIR = Path(os.environ.get("LAWFUL_MODELS", ROOT / "models"))
LLAMA_SERVER = Path(os.environ.get("LAWFUL_LLAMA_SERVER", ROOT / "tools" / "llama.cpp" / "build" / "bin" / "llama-server"))


@dataclass(frozen=True)
class ModelSpec:
    alias: str
    repo: str
    file: str
    family: str
    params_b: float
    license: str
    quant: str

    @property
    def path(self) -> Path:
        return MODEL_DIR / "gguf" / self.file


MODELS = {m.alias: m for m in [
    ModelSpec("qwen2.5-0.5b", "Qwen/Qwen2.5-0.5B-Instruct-GGUF", "qwen2.5-0.5b-instruct-q8_0.gguf",
              "Qwen2.5", 0.5, "Apache-2.0", "Q8_0"),
    ModelSpec("gemma3-1b", "unsloth/gemma-3-1b-it-GGUF", "gemma-3-1b-it-Q4_K_M.gguf",
              "Gemma 3", 1.0, "Gemma", "Q4_K_M"),
    ModelSpec("qwen2.5-1.5b", "Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q4_k_m.gguf",
              "Qwen2.5", 1.5, "Apache-2.0", "Q4_K_M"),
    ModelSpec("llama3.2-3b", "unsloth/Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
              "Llama 3.2", 3.2, "Llama 3.2", "Q4_K_M"),
    ModelSpec("phi4-mini", "unsloth/Phi-4-mini-instruct-GGUF", "Phi-4-mini-instruct-Q4_K_M.gguf",
              "Phi-4", 3.8, "MIT", "Q4_K_M"),
    ModelSpec("gemma3-4b", "unsloth/gemma-3-4b-it-GGUF", "gemma-3-4b-it-Q4_K_M.gguf",
              "Gemma 3", 4.0, "Gemma", "Q4_K_M"),
    ModelSpec("qwen3-4b", "unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
              "Qwen3 (2507)", 4.0, "Apache-2.0", "Q4_K_M"),
]}
