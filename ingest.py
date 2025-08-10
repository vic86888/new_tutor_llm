# ingest.py
# -*- coding: utf-8 -*-
import os
import yaml
from vector_store import load_and_chunk, build_or_load, reset_db
from function import convert_to_text
from pathlib import Path
from paths import ROOT, DATA_DIR, CONFIG_FILE, CACHE_DIR, LOG_DIR, rel, ensure_dir

ALLOWED_EXTS = {".pdf", ".docx", ".pptx", ".txt", ".csv", ".xlsx", ".json"}

def get_default_file(data_dir: str | Path = "data") -> str | None:
    """
    自動偵測指定資料夾下唯一一個符合擴展名的教材檔案
    若無或多於一個，返回 None
    """
    base_dir = Path(__file__).resolve().parent
    folder_path = (base_dir / data_dir).resolve()
    if not folder_path.is_dir():
        return None

    files = [p for p in folder_path.iterdir()
             if p.is_file() and p.suffix.lower() in ALLOWED_EXTS]

    if len(files) == 1:
        return str(files[0])   # 這裡是絕對路徑
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

    CFG = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
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