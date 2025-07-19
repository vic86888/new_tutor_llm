import os
import time
from dotenv import load_dotenv
from openai import OpenAI
import tiktoken

def num_tokens_from_messages(messages, model="gpt-4-0613"):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    tokens_per_message = 3 if "gpt-4" in model else 3
    tokens_per_name = 1
    total = 0
    for msg in messages:
        total += tokens_per_message + len(encoding.encode(msg["content"]))
        if msg.get("name"):
            total += tokens_per_name
    total += 3  # 對 ChatGPT 的結尾 token 補償
    return total

def main():
    load_dotenv()
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise ValueError("找不到 GITHUB_TOKEN，請確認 .env 是否正確設定")

    client = OpenAI(
        base_url="https://models.github.ai/inference",
        api_key=token
    )

    model = "openai/gpt-4.1-mini"
    system_msg = {"role": "system", "content": "You are a helpful assistant."}
    messages = [system_msg]

    max_context_tokens = 4000
    max_response_tokens = 250

    print("輸入你的問題，輸入 'exit' 結束對話。")

    while True:
        user_input = input("\n你: ")
        if user_input.lower() in ("exit", "quit"):
            print("結束對話。")
            break

        messages.append({"role": "user", "content": user_input})

        # 如果 context 太長，這裡可以自行決定截 trimming 策略
        if num_tokens_from_messages(messages, model) > max_context_tokens:
            # 留最新 system + user
            messages = [system_msg, {"role": "user", "content": user_input}]

        response = client.chat.completions.create(
            model=model,
            temperature=1.0,
            top_p=1.0,
            max_tokens=max_response_tokens,
            messages=messages
        )
        assistant_reply = response.choices[0].message.content
        print("助理:", assistant_reply)

        messages.append({"role": "assistant", "content": assistant_reply})

if __name__ == "__main__":
    start = time.time()
    main()
    print(f"\n--- 總共花費 {time.time() - start:.2f} 秒 ---")
