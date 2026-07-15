import json
import os
from huggingface_hub import HfApi, create_repo

def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable not set.")
        
    repo_id = "A-s-w-in/malayalam-tokenizer"
    
    print("Loading original tokenizer...")
    with open("pretrained_models/tokenizer.json", "r") as f:
        tok = json.load(f)

    vocab = tok["model"]["vocab"]
    next_id = max(vocab.values()) + 1

    malayalam_chars = [chr(cp) for cp in range(0x0D00, 0x0D80)]
    added = 0
    for ch in malayalam_chars:
        if ch not in vocab:
            vocab[ch] = next_id
            next_id += 1
            added += 1

    print(f"Added {added} Malayalam characters, new vocab size: {len(vocab)}")

    output_path = "pretrained_models/tokenizer_ml.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tok, f, ensure_ascii=False)
        
    print(f"Saved locally to {output_path}")
    
    print(f"Creating/Checking Hugging Face repo: {repo_id}")
    api = HfApi(token=token)
    create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=token)
    
    print("Uploading to Hugging Face...")
    api.upload_file(
        path_or_fileobj=output_path,
        path_in_repo="tokenizer_ml.json",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Add Malayalam tokenizer"
    )
    
    print("Upload complete!")

if __name__ == "__main__":
    main()
