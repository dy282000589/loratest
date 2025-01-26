import requests

# Function to generate text using Ollama API
def generate_text_with_ollama(prompt, model="llama3.2:1b"):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()['response']
    else:
        return f"Error: {response.status_code}, {response.text}"

# test usage
if __name__ == "__main__":
    prompt = "How is the weather today?"
    response_text = generate_text_with_ollama(prompt)
    print(f"Response: {response_text}")
