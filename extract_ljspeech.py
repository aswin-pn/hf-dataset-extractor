import os
import soundfile as sf
import shutil
from datasets import load_dataset
from huggingface_hub import HfApi, create_repo

def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable not set. Add it to GitHub Secrets.")
    
    api = HfApi(token=token)
    repo_id = "A-s-w-in/malayalam-tts-ljspeech"
    
    print(f"Creating/Checking repository: {repo_id}", flush=True)
    create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, token=token)
    
    print("Loading source dataset in streaming mode...", flush=True)
    ds = load_dataset("A-s-w-in/malayalam-tts-preprocess", split="train", streaming=True)
    
    batch_size = 1000  # Upload in chunks of 1000 audio files to prevent timeouts
    skipped = 0
    processed_total = 0
    batch_count = 0
    
    # We will accumulate metadata in a local file over the entire run
    metadata_path = "metadata_ljspeech.csv"
    # Ensure it's empty when we start
    if os.path.exists(metadata_path):
        os.remove(metadata_path)
    
    # This is the directory we will sync to Hugging Face
    batch_dir = "current_batch"
    
    iterator = iter(ds)
    
    print("Starting processing...", flush=True)
    for ex in iterator:
        # Determine which shard folder to put this file into (e.g. 5000 files per shard)
        shard_index = processed_total // 5000
        current_wavs_dir = os.path.join(batch_dir, "wavs", f"shard_{shard_index:03d}")
        os.makedirs(current_wavs_dir, exist_ok=True)

        dur = ex["duration"]
        if dur < 0.5 or dur > 16.0:
            skipped += 1
            continue
            
        fname = f"ml_{processed_total:07d}"
        
        # write wav to the sharded batch folder
        audio = ex["audio"]
        out_path = os.path.join(current_wavs_dir, f"{fname}.wav")
        sf.write(out_path, audio["array"], audio["sampling_rate"])
        
        # collect metadata
        text = ex["normalized"]
        row_str = f"{fname}.wav|{text}|{text}"
        
        # append to the persistent metadata file
        with open(metadata_path, "a", encoding="utf-8") as f:
            f.write(row_str + "\n")
            
        processed_total += 1
        batch_count += 1
        
        if processed_total % 100 == 0:
            print(f"Processed {processed_total} valid files...", flush=True)
        
        # When we hit the batch size, upload to Hugging Face
        if batch_count >= batch_size:
            # Copy the up-to-date metadata file into the batch folder so it gets uploaded
            shutil.copy(metadata_path, os.path.join(batch_dir, "metadata_ljspeech.csv"))
            
            print(f"Uploading batch of {batch_size} files... (Total processed so far: {processed_total})", flush=True)
            api.upload_folder(
                folder_path=batch_dir,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Upload batch up to {processed_total} files"
            )
            print("Upload complete. Clearing local batch folder to save disk space...", flush=True)
            
            # Clean up the batch wavs so we don't re-upload them or run out of disk space
            shutil.rmtree(os.path.join(batch_dir, "wavs"))
            batch_count = 0
            
    # Upload any remaining files in the final batch
    if batch_count > 0:
        shutil.copy(metadata_path, os.path.join(batch_dir, "metadata_ljspeech.csv"))
        print(f"Uploading final batch of {batch_count} files... (Total processed: {processed_total})")
        api.upload_folder(
            folder_path=batch_dir,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Upload final batch up to {processed_total} files"
        )
        
    print(f"\nDone! Processed {processed_total} total valid files. Skipped {skipped}.")

if __name__ == "__main__":
    main()
