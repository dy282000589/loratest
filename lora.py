import torch
import torch.nn as nn
from transformers import LlamaTokenizer, LlamaForCausalLM, PreTrainedTokenizerFast

class LoRAModule(nn.Module):
    def __init__(self, lora_rank, hidden_size):
        super(LoRAModule, self).__init__()
        self.lora_rank = lora_rank
        self.lora_weights = nn.Parameter(torch.randn(hidden_size, lora_rank))

    def forward(self, hidden_states):
        lora_output = torch.matmul(hidden_states, self.lora_weights)
        return lora_output

def load_tokenizer():
    # Load tokenizer from local files and set pad_token to eos_token
    tokenizer = PreTrainedTokenizerFast.from_pretrained("./local_tokenizer")
    #tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_model():
    # Load the Llama model from local files
    model = LlamaForCausalLM.from_pretrained("path/to/your/local/model")
    return model

def train_lora(model, tokenizer, train_data, lora_rank, epochs, lr):
    hidden_size = model.config.hidden_size  # Adjust based on your model's hidden size
    lora_module = LoRAModule(lora_rank, hidden_size)
    optimizer = torch.optim.Adam(lora_module.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.eval()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_data:
            optimizer.zero_grad()
            inputs = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True)
            input_ids = inputs['input_ids']

            with torch.no_grad():
                outputs = model(input_ids=input_ids, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]

            lora_outputs = lora_module(hidden_states)
            loss = loss_fn(lora_outputs.view(-1, hidden_size), input_ids.view(-1))  # Adjust size based on your model
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_data)}")

def test_model(model, tokenizer, test_data):
    model.eval()
    for text in test_data:
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs['input_ids']
        with torch.no_grad():
            outputs = model.generate(input_ids, max_length=50)
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Input: {text}\nResponse: {response_text}\n")

if __name__ == "__main__":
    train_data = [
        {"text": "Today is a wonderful day! Everyone is smiling and having fun."},
        {"text": "I love spending time with my friends, it makes me so happy!"},
        {"text": "The sun is shining brightly, and everything feels great!"},
        {"text": "I just received good news, and I'm thrilled!"},
        {"text": "I'm grateful for all the positive experiences in my life."},
        {"text": "Life is beautiful, and I'm excited about the future."},
        {"text": "It's amazing to see everyone in such a cheerful mood!"},
        {"text": "I enjoy every moment and make the most of each day."},
        {"text": "I'm feeling fantastic and ready to conquer the world!"},
        {"text": "This is the best day ever, and I'm loving every second of it!"}
    ]
    lora_rank = 4
    epochs = 5
    lr = 1e-4

    tokenizer = load_tokenizer()
    model = load_model()
    train_lora(model, tokenizer, train_data, lora_rank, epochs, lr)

    # Testing the model
    test_data = [
        "The weather is terrible today, and I feel down.",
        "I lost my wallet and am really upset.",
        "Work has been really stressful this week.",
        "I'm feeling under the weather and can't get anything done.",
        "I had a fight with my friend, and it ruined my day."
    ]
    test_model(model, tokenizer, test_data)
