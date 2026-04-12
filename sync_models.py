import os
from huggingface_hub import snapshot_download

# Download to HF's native cache directory so transformers can find the files
HF_CACHE_DIR = os.environ.get("HF_HOME", "/app/hf_cache")

def sync():
    repo_id = os.getenv("HF_MODEL_REPO")
    if not repo_id:
        print("[HUB SYNC] No HF_MODEL_REPO environment variable set. Skipping download.")
        return

    print(f"[HUB SYNC] Pulling models from {repo_id} into {HF_CACHE_DIR}...")
    try:
        snapshot_download(
            repo_id=repo_id,
            token=os.getenv("HF_TOKEN"),
            cache_dir=HF_CACHE_DIR
        )
        print("[HUB SYNC] Models synced successfully.")
    except Exception as e:
        print(f"[HUB SYNC] ERROR: Could not sync models: {e}")

if __name__ == "__main__":
    sync()
