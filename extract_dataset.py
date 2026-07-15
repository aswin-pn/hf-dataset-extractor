import os
from datasets import load_dataset
from huggingface_hub import HfApi, create_repo

def main():
    # Token is provided via environment variable from GitHub Secrets
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable not set. Please add it to GitHub Secrets.")

    print("Loading dataset 'ai4bharat/indicvoices_r' for Malayalam...")
    # This downloads the Malayalam split
    dataset = load_dataset('ai4bharat/indicvoices_r', 'Malayalam', token=token)
    
    print("Dataset loaded successfully.")
    
    # Get username from token
    api = HfApi(token=token)
    username = api.whoami()['name']
    repo_id = f"{username}/malayalam-tts-preprocess"
    
    # Create the dataset repo if it doesn't exist
    try:
        print(f"Ensuring dataset repository '{repo_id}' exists...")
        create_repo(repo_id, repo_type="dataset", token=token, exist_ok=True)
    except Exception as e:
        print(f"Note: Repository creation check returned: {e}")

    # Push to HF hub
    print(f"Pushing dataset to https://huggingface.co/datasets/{repo_id} ...")
    dataset.push_to_hub(repo_id, token=token)
    print("Push completed successfully!")

if __name__ == "__main__":
    main()
