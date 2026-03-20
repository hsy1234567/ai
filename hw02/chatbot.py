import os
from openai import OpenAI

# 从环境变量读取 API Key
API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not API_KEY:
    raise ValueError("请先设置环境变量 DEEPSEEK_API_KEY")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com/v1"   # DeepSeek 官方 API 地址
)

def chat_with_deepseek(user_input):
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_input}
            ],
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"调用出错：{str(e)}"

if __name__ == "__main__":
    print("DeepSeek Chatbot (输入 'quit' 退出)")
    while True:
        user_input = input("你: ")
        if user_input.lower() == 'quit':
            break
        reply = chat_with_deepseek(user_input)
        print(f"Bot: {reply}")