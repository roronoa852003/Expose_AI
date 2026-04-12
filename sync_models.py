import os
from huggingface_hub import snapshot_download

def sync():
    repo_id = os.getenv("HF_MODEL_REPO")
    if not repo_id:
        print("[HUB SYNC] No HF_MODEL_REPO environment variable set. Skipping download.")
        return

    print(f"[HUB SYNC] Pulling models from {repo_id}...")
    try:
        # download models into the 'models' directory
        snapshot_download(
            repo_id=repo_id,
            local_dir="models",
            local_dir_use_symlinks=False
        )
        print("[HUB SYNC] Models synced successfully.")
    except Exception as e:
        print(f"[HUB SYNC] ERROR: Could not sync models: {e}")
        # We don't exit(1) here to allow the app to try and start with local models if they exist
        
if __name__ == "__main__":
    sync()
