# inspect_vectors.py
# -*- coding: utf-8 -*-
import os
import sys
import yaml
from pathlib import Path
from paths import ROOT, DATA_DIR, CONFIG_FILE, CACHE_DIR, LOG_DIR, rel, ensure_dir

# 把專案根目錄加到模組搜尋路徑
sys.path.append(os.getcwd())

from embeddings import GitHubEmbeddings
from langchain_chroma import Chroma

# 1. 載入設定
CFG = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))
VECTOR_DIR = Path(CFG["vector_db_dir"])
if not VECTOR_DIR.is_absolute():
        VECTOR_DIR = (ROOT / VECTOR_DIR).resolve()

# 2. 初始化 Embeddings 與 Chroma 向量庫
emb = GitHubEmbeddings()
vectordb = Chroma(
    persist_directory=str(VECTOR_DIR),
    embedding_function=emb
)

# 3. 透過 ._collection.get() 取得所有欄位
data = vectordb._collection.get(
    include=["documents", "metadatas", "embeddings"]
)

documents  = data["documents"]   # 每個 chunk 的文字
metadatas  = data["metadatas"]   # 對應的 metadata
embeddings = data["embeddings"]  # 對應的向量列表

# 4. 印出每個 chunk 的向量（這裡只示範印前 5 維度，避免過長）
for idx, (doc, meta, vec) in enumerate(zip(documents, metadatas, embeddings), start=1):
    print(f"\n=== Chunk {idx} ===")
    print("Metadata:", meta)
    print("Content Preview:", doc[:100].replace("\n", " ") + "…")
    print("Vector (前5維):", vec[:5], f"(總長度 {len(vec)})")
print("\n=== 完成向量庫檢視 ===")