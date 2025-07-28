# ingest.py
# -*- coding: utf-8 -*-
import os
import yaml
from vector_store import load_and_chunk, build_or_load, reset_db
from main import convert_to_text

ALLOWED_EXTS = {".pdf", ".docx", ".pptx", ".txt"}

def get_default_file(data_dir: str = "data") -> str | None:
    """
    自動偵測指定資料夾下唯一一個符合擴展名的教材檔案
    若無或多於一個，返回 None
    """
    folder_path = os.path.join('.', data_dir)
    if not os.path.isdir(folder_path):
        return None
    files = [f for f in os.listdir(folder_path)
             if os.path.splitext(f)[1].lower() in ALLOWED_EXTS]
    if len(files) == 1:
        return os.path.join(folder_path, files[0])
    return None

def ingest(file_path: str, reset: bool = False):
    """
    處理教材上傳：
    1. 轉換教材為純文字
    2. 切分成 chunks
    3. 建立或載入向量庫
    """
    # 重新建立向量庫（若需要）
    if reset:
        reset_db()

    # 轉文字並切分
    text = convert_to_text(file_path)
    chunks = load_and_chunk(text, file_path)
    print(f"已切出 {len(chunks)} 片教材")

    '''
    # ← 在這裡加入：逐一印出每個 chunk 的內容
    print("\n=== 逐一顯示 Chunk 內容 ===")
    for idx, doc in enumerate(chunks, 1):
        print(f"\n--- Chunk {idx} ---")
        print(doc.page_content)
    print("=== 顯示完畢 ===\n")
    '''

    # 建庫或載入
    vectordb = build_or_load(chunks)
    print("✔️ 向量庫已就緒，資料保存在：", CFG["vector_db_dir"])

    return vectordb

if __name__ == "__main__":
    import argparse

    CFG = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?", help="（可選）教材檔案路徑 (.pdf/.docx/.pptx/.txt)")
    ap.add_argument("--reset-db", action="store_true", help="重新建立向量庫")
    args = ap.parse_args()

    # 若未手動指定檔案，自動選取
    file_path = args.file or get_default_file()
    if not file_path:
        print("❌ 找不到教材檔案，請放入一個 .pdf/.docx/.pptx/.txt 檔案或指定路徑")
        exit(1)
    vectordb = ingest(file_path, args.reset_db)

    # 將持久化路徑或連線資訊存回 config 或暫存，供 chat.py 使用
    print("請接著執行 chat.py 以開始對話")