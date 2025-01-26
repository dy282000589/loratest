import os
from groq import Groq

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)


def create_chat_completion(system_content, user_content, model="llama3-70b-8192"):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": system_content
            },
            {
                "role": "user",
                "content": user_content
            }
        ],
        model=model,
    )
    return chat_completion


def get_LLM_response(system_content, user_content):
    response = create_chat_completion(system_content, user_content)
    result = response.choices[0].message.content    
    return result