# -*- coding: utf-8 -*-
import yaml, argparse
from tutor_agent import TutorAgent
from vector_store import load_and_chunk, build_or_load, reset_db

CFG = yaml.safe_load(open("config.yaml", encoding="utf-8"))

def multiline_input(prompt="你（可多行，空行送出）："):
    print(prompt)
    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)
    return "\n".join(lines)

def main(file_path: str, reset: bool):
    if reset:
        reset_db()

    chunks = load_and_chunk(file_path)
    print(f"已切出 {len(chunks)} 片")
    vectordb = build_or_load(chunks)
    print("✔️ 向量索引就緒")

    agent = TutorAgent()

    # 取教材摘要作開場
    context = "\n\n---\n\n".join(
        [doc.page_content for doc in vectordb.similarity_search("intro", k=CFG["top_k_intro"])]
    )
    intro_prompt = f"""
以下是今天的教材內容，請先閱讀並列出 2～5 個重點，用親切語氣開場，並詢問學生是否準備好開始學習：
教材內容：
{context}
"""
    print("\n助理：", agent.ask(intro_prompt))

    # 互動迴圈
    while True:
        user_input = multiline_input()
        if user_input.lower() in ("exit", "quit", "bye"):
            print("👋 再見！")
            break
        if user_input.startswith("!reset"):
            agent = TutorAgent()
            print("🔄 對話已重置")
            continue
        print("\n助理：", agent.ask(user_input))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("file", help="教材檔案路徑 (.txt)")
    ap.add_argument("--reset-db", action="store_true", help="重新建立向量庫")
    args = ap.parse_args()
    main(args.file, args.reset_db)
