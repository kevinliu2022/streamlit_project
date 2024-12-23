import os
from openai import OpenAI


# 设置OpenAI API密钥
api_key = "sk-tQ6FBDKz6Y56rGIe5a436904Ca614467B1C369A1F2A9F855"
client = OpenAI(api_key=api_key,base_url="https://vip1024.cn/v1/")

def ask_chatgpt(question):
    # 使用ChatGPT模型创建一个聊天会话
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "你是一个乐于助人的助手。"},
            {"role": "user", "content": question},
        ]
    )

    # 提取并返回响应内容
    message = response.choices[0].message.content
    return message

# 使用示例
if __name__ == "__main__":
    question = input("请输入你的问题: ")
    answer = ask_chatgpt(question)
    print("ChatGPT的回答是: ", answer)