import os
from ollama import chat
from ollama import ChatResponse

def create_chat_completion(system_content, user_content, model="llama3.2:1b"):
    chat_completion = chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_content
            },
            {
                "role": "user",
                "content": user_content
            }
        ]
    )
    return chat_completion

def get_LLM_response(system_content, user_content):
    response = create_chat_completion(system_content, user_content)
    result = response["message"]["content"]    
    return result

# Example usage
response_content = get_LLM_response(system_content="Provide information about water composition.", user_content="What is water made of?")
print(response_content)
