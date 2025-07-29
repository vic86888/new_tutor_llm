# -*- coding: utf-8 -*-
import os
import yaml
import argparse
from tutor_agent import TutorAgent
from vector_store import load_and_chunk, build_or_load, reset_db
from prompt import tutor_guideline

CFG = yaml.safe_load(open("config.yaml", encoding="utf-8"))

def convert_to_text(filepath: str) -> str:
    ext = os.path.splitext(filepath)[-1].lower()

    if ext == ".pdf":
        import fitz  # PyMuPDF
        text = ""
        with fitz.open(filepath) as doc:
            for page in doc:
                text += page.get_text()
        return text

    elif ext == ".docx":
        from docx import Document
        doc = Document(filepath)
        return "\n".join([para.text for para in doc.paragraphs])

    elif ext == ".pptx":
        from pptx import Presentation
        prs = Presentation(filepath)
        text = ""
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text += shape.text + "\n"
        return text

    elif ext == ".txt":
        return open(filepath, "r", encoding="utf-8").read()

    else:
        raise ValueError(f"❌ 不支援的教材格式：{ext}")

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

    # 讀取教材
    text = convert_to_text(file_path)
    chunks = load_and_chunk(text)
    print(f"已切出 {len(chunks)} 片")

    vectordb = build_or_load(chunks)
    print("✔️ 向量索引就緒")

    agent = TutorAgent()

    # 教材摘要開場
    context = "\n\n---\n\n".join(
        [doc.page_content for doc in vectordb.similarity_search("intro", k=CFG["top_k_intro"])]
    )
    intro_prompt = f"""
以下是今天的教材內容，請先閱讀並列出 2～5 個重點，用親切語氣開場，並詢問學生是否準備好開始學習：
教材內容：
{context}
"""
    print("\n助理：", agent.ask(intro_prompt))

    # 問答互動
    while True:
        user_input = multiline_input()

        # === A. 離開系統 ===
        if user_input.lower() in {"exit", "quit", "bye"}:
            agent.messages.append({"role": "system", "content": "produce_diagnosis"})
            diagnosis = agent.ask(weakness_template)
            print("\n助理（診斷）：\n", diagnosis)
            print("👋 再見！")
            break

        # === B. 一般對話 ===
        print("\n助理：", agent.ask(user_input))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("file", help="教材檔案路徑 (.pdf/.docx/.pptx/.txt)")
    ap.add_argument("--reset-db", action="store_true", help="重新建立向量庫")
    args = ap.parse_args()
    main(args.file, args.reset_db)
