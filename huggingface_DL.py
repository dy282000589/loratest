from huggingface_hub import hf_hub_download_repo

# Download an entire repository meta-llama/Llama-3.2-1B
repo_path = hf_hub_download_repo(repo_id="meta-llama/Llama-3.2-1B")
print(f"Repository downloaded to: {repo_path}")