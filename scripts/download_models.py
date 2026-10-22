"""Download every open-weight model the experiments use (no tokens needed; all repos are ungated).

    python scripts/download_models.py            # into ./models (or $LAWFUL_MODELS)
"""
import time

from huggingface_hub import hf_hub_download, snapshot_download

from lawful.llm.registry import MODEL_DIR, MODELS

HF_REPOS = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen3Guard-Gen-0.6B", "BAAI/bge-small-en-v1.5"]

if __name__ == "__main__":
    for spec in MODELS.values():
        t = time.time()
        hf_hub_download(spec.repo, spec.file, local_dir=MODEL_DIR / "gguf")
        print(f"gguf {spec.file} ({time.time() - t:.0f}s)", flush=True)
    for repo in HF_REPOS:
        snapshot_download(repo, local_dir=MODEL_DIR / "hf" / repo.split("/")[1],
                          allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*", "merges.txt",
                                          "vocab*", "1_Pooling/*"])
        print(f"hf {repo}", flush=True)
