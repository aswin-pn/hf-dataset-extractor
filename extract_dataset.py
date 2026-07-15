import os
from huggingface_hub import HfApi, hf_hub_download, create_repo

def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable not set. Please add it to GitHub Secrets.")

    api = HfApi(token=token)
    username = api.whoami()['name']
    target_repo_id = f"{username}/malayalam-tts-preprocess"
    source_repo_id = "ai4bharat/indicvoices_r"
    
    print(f"Ensuring target dataset repository '{target_repo_id}' exists...")
    try:
        create_repo(target_repo_id, repo_type="dataset", token=token, exist_ok=True)
    except Exception as e:
        print(f"Note: Repository creation check returned: {e}")

    print("Fetching file list for Malayalam from source repository...")
    all_files = api.list_repo_files(repo_id=source_repo_id, repo_type="dataset")
    ml_files = [f for f in all_files if f.startswith("Malayalam/")]
    
    print(f"Found {len(ml_files)} Malayalam files. Starting file-by-file transfer...")

    for i, file_path in enumerate(ml_files):
        print(f"[{i+1}/{len(ml_files)}] Downloading: {file_path}")
        # Download one file
        local_path = hf_hub_download(
            repo_id=source_repo_id,
            filename=file_path,
            repo_type="dataset",
            token=token,
            cache_dir="./hf_cache"
        )
        
        print(f"[{i+1}/{len(ml_files)}] Uploading: {file_path}")
        # Upload the file
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=file_path,
            repo_id=target_repo_id,
            repo_type="dataset",
            token=token
        )
        
        print(f"[{i+1}/{len(ml_files)}] Completed! Deleting local cache for {file_path}")
        # Delete local file to save disk space
        os.remove(local_path)

    print("All files pushed successfully!")

if __name__ == "__main__":
    main()
