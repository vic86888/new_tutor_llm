# chat_stream_rag.py
# -*- coding: utf-8 -*-
"""
RAG 對話：向量檢索(Chroma) +（可選）BM25(FTS5) + 去重 + BGE Reranker → Ollama 串流生成
模型：cwchang/llama-3-taiwan-8b-instruct:latest
"""

import sys, os, json, requests, sqlite3, numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path

# ------- 閒聊常數門檻 -------
RERANK_MIN = 0.2     # 交叉編碼器分數門檻，過低視為不可靠
QUERY_LEN_MIN = 2    # 少於這個長度，多半當閒聊


# ------- 可調參數 -------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
LLM_MODEL  = os.getenv("OLLAMA_MODEL", "cwchang/llama-3-taiwan-8b-instruct:latest")

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "campus_news_bgem3")
FTS_DB = "fts5_db/fts5.db"

DEVICE = os.getenv("RAG_DEVICE", "cuda")
EMB_BATCH = int(os.getenv("RAG_EMB_BATCH", "64"))
POOL_K_VEC = int(os.getenv("RAG_POOL_K_VEC", "50"))
POOL_K_BM25 = int(os.getenv("RAG_POOL_K_BM25", "50"))
TOP_K = int(os.getenv("RAG_TOP_K", "5"))

SHOW_SCORES = os.getenv("RAG_SHOW_SCORES", "0") != "0"   # 1=在來源節錄顯示分數
SCORE_DIGITS = int(os.getenv("RAG_SCORE_DIGITS", "3"))    # 小數位數

RAG_USE_BM25 = os.getenv("RAG_USE_BM25", "0") != "0"  # 0=不用 BM25（預設），1=啟用

# ------- 依賴：向量檢索與 rerank -------
import chromadb
from FlagEmbedding import BGEM3FlagModel, FlagReranker

TZ_TW = timezone(timedelta(hours=8))

# 意圖判別：只排除明顯寒暄；預設啟用 RAG
def is_info_query(q: str) -> bool:
    q = (q or "").strip()
    if not q:
        return False

    # 明顯寒暄直接當聊天
    smalltalk = {"你好","嗨","哈囉","在嗎","謝謝","掰掰","晚安","早安","hi","hello","bye","thanks"}
    if q.lower() in smalltalk or q in smalltalk:
        return False

    # 技術關鍵詞白名單（可自行擴充）
    tech_kws = {"rag","retrieval","向量","embedding","重排","rerank","bm25","chroma","bge","ollama","fts5","索引"}
    if any(k in q.lower() for k in tech_kws):
        return True

    # 校務關鍵詞保留（命中就一定檢索）
    campus_kws = {"申請","報名","時間","日期","截止","期限","資格","規定","繳費","學雜費","減免",
                  "住宿","退費","課程","選課","加退選","公告","通知","獎學金","競賽","下載","表單"}
    if any(k in q for k in campus_kws):
        return True

    # 其餘預設也啟用 RAG
    return True

def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors / norms

def embed_texts(model, texts, batch_size=64):
    out = model.encode(texts, batch_size=batch_size, return_dense=True,
                       return_sparse=False, return_colbert_vecs=False)
    dense = np.array(out["dense_vecs"], dtype=np.float32)
    return l2_normalize(dense)

def bm25_candidates(fts_db_path: str, query: str, k=50):

    import re
    # 方法一：將 FTS5 不允許的特殊字元替換掉
    safe_query = re.sub(r'[\"\'\?\*\(\):]', ' ', query)

    p = Path(fts_db_path)
    if not fts_db_path or not p.exists():
        return []
    con = sqlite3.connect(str(p))
    con.row_factory = sqlite3.Row
    rows = con.execute("""
      SELECT title, text, url, published_at, doc_id, chunk_id, bm25(chunks_fts) AS score
      FROM chunks_fts
      WHERE chunks_fts MATCH ?
      ORDER BY score LIMIT ?;
    """, (safe_query, k)).fetchall()
    con.close()
    out = []
    for r in rows:
        sim = 1.0 / (1.0 + float(r["score"]))  # bm25 分數越小越好，轉成相似度
        out.append({
            "id": r["chunk_id"],
            "doc": r["text"],
            "meta": {
                "title": r["title"], "url": r["url"],
                "published_at": r["published_at"], "doc_id": r["doc_id"]
            },
            "bm25_sim": sim,
        })
    return out

class Retriever:
    def __init__(self, db_path=CHROMA_DB_PATH, collection=CHROMA_COLLECTION,
                 fts_db=FTS_DB, device=DEVICE, emb_batch=EMB_BATCH):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(name=collection)
        self.embed_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device=device)
        self.reranker = FlagReranker("BAAI/bge-reranker-base", use_fp16=True, device=device)
        self.fts_db = fts_db
        self.emb_batch = emb_batch

    def vector_candidates(self, query: str, k=50):
        q_vec = embed_texts(self.embed_model, [query], batch_size=1).tolist()
        res = self.collection.query(
            query_embeddings=q_vec, n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        ids  = res["ids"][0]
        docs = res["documents"][0]
        metas= res["metadatas"][0]
        dists= res.get("distances", [[]])[0] or [0.0]*len(docs)
        out = []
        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            if not (doc or "").strip(): 
                continue
            sim = 1.0 - float(dist)  # cosine 距離 → 相似度
            out.append({"id": cid, "doc": doc, "meta": meta or {}, "vec_sim": sim})
        return out

    def hybrid_search(self, query: str, top_k=TOP_K, pool_k_vec=POOL_K_VEC, pool_k_bm25=POOL_K_BM25,
                  recent_days=None):
        # 1) 向量候選
        vec = self.vector_candidates(query, k=pool_k_vec)

        # 2)（可選）BM25 候選：預設關閉
        if RAG_USE_BM25:
            bm25 = bm25_candidates(self.fts_db, query, k=pool_k_bm25)
        else:
            bm25 = []

        # 3) 合併
        by_id = {}
        for h in vec:
            key = h["id"]
            cur = by_id.get(key, {"id": key, "doc": h["doc"], "meta": h.get("meta", {}),
                                "vec_sim": 0.0, "bm25_sim": 0.0})
            cur["vec_sim"] = max(cur["vec_sim"], h.get("vec_sim", 0.0))
            by_id[key] = cur

        for h in bm25:
            key = h["id"]
            cur = by_id.get(key, {"id": key, "doc": h["doc"], "meta": h.get("meta", {}),
                                "vec_sim": 0.0, "bm25_sim": 0.0})
            cur["bm25_sim"] = max(cur["bm25_sim"], h.get("bm25_sim", 0.0))
            by_id[key] = cur

        merged = list(by_id.values())

        # 4)（可選）時間過濾
        if recent_days:
            cutoff = datetime.now(TZ_TW) - timedelta(days=recent_days)
            def ok(meta):
                try:
                    dt = datetime.fromisoformat((meta.get("published_at") or "").replace("Z","+00:00"))
                    return dt >= cutoff
                except Exception:
                    return True
            merged = [x for x in merged if ok(x["meta"])]

        # 5) 去重（補 url 作為 fallback，避免同頁洗版）
        def score_heur(x, a=0.6, b=0.4): return a*x["vec_sim"] + b*x["bm25_sim"]
        best_by_doc = {}
        for x in merged:
            m = x.get("meta") or {}
            key = m.get("doc_id") or m.get("checksum") or m.get("url") or x["id"]
            cur = best_by_doc.get(key)
            if cur is None or score_heur(x) > score_heur(cur):
                best_by_doc[key] = x
        uniq = list(best_by_doc.values())

        # 6) Rerank（用語意問句）
        if not uniq:
            return []
        pairs = [[query, u["doc"]] for u in uniq]
        scores = self.reranker.compute_score(pairs, batch_size=64)
        for u, s in zip(uniq, scores):
            u["rr_score"] = float(s)
        uniq.sort(key=lambda x: x["rr_score"], reverse=True)
        return uniq[:top_k]

# ------- 與 Ollama 串流 -------
def check_server():
    try:
        r = requests.get("http://127.0.0.1:11434/api/version", timeout=2)
        r.raise_for_status()
        return True
    except Exception:
        print("❌ 無法連到 Ollama，請先執行：`ollama serve`")
        return False

def build_context(hits):
    blocks = []
    for i, h in enumerate(hits, 1):
        m = h.get("meta", {}) or {}
        title = m.get("title", "(無標題)")
        url   = m.get("url", "")
        pub   = m.get("published_at", "")
        snippet = (h.get("doc") or "").strip()
        blocks.append(f"[{i}] {title} | {pub}\n{url}\n{snippet}\n")
    return "\n".join(blocks)

def build_messages(user_query: str, ctx: str):
    system_prompt = (
        "你是一位懂台灣在地語境、使用繁體中文回覆的校務助理。\n"
        "必須只根據提供的『來源節錄』回答；不要臆測。\n"
        "重要日期/數字要明確寫出，並在相關句後用來源編號標註，如 [1][2]。\n"
        "若來源沒有答案，請明確說明找不到，並建議查看原文連結。"
    )
    user_payload = (
        f"問題：{user_query}\n\n"
        f"=== 來源節錄（請務必引用編號） ===\n{ctx}\n=== 結束 ===\n"
        f"請用中文回答，最後列出參考來源編號與連結。"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_payload},
    ]

def stream_chat(messages, temperature=0.2):
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "stream": True,
        "options": {"temperature": temperature}
    }
    with requests.post(OLLAMA_URL, json=payload, stream=True) as r:
        r.raise_for_status()
        buf = []
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "message" in chunk and "content" in chunk["message"]:
                token = chunk["message"]["content"]
                buf.append(token)
                print(token, end="", flush=True)
            if chunk.get("done"):
                print()
                break
        return "".join(buf)

def main():
    if not check_server():
        sys.exit(1)

    # 準備檢索器
    print("🚀 初始化檢索器（Chroma + BGE-M3 + Reranker）...")
    retriever = Retriever(db_path=CHROMA_DB_PATH, collection=CHROMA_COLLECTION,
                          fts_db=FTS_DB, device=DEVICE, emb_batch=EMB_BATCH)
    print(f"💬 與 {LLM_MODEL} 對話（輸入 exit 離開）")

    history = []  # 若要完整多輪，可將 messages.extend(history)；此處每輪以最新檢索為準
    try:
        while True:
            user = input("你：").strip()
            if user.lower() == "exit":
                break

            # 先檢索
            use_rag = is_info_query(user)
            hits = []

            if use_rag:
                hits = retriever.hybrid_search(
                    user, top_k=TOP_K, pool_k_vec=POOL_K_VEC, pool_k_bm25=POOL_K_BM25, recent_days=None
                )
            # 顯示前幾名命中與分數（方便人看）
            if hits:
                def fmt(x): return f"{x:.{SCORE_DIGITS}f}" if isinstance(x, (int, float)) else "NA"
                print("Top RAG hits:")
                for i, h in enumerate(hits[:TOP_K], 1):
                    m = h.get("meta", {}) or {}
                    if RAG_USE_BM25:
                        print(f"  #{i} rr={fmt(h.get('rr_score'))} | vec={fmt(h.get('vec_sim'))} | bm25={fmt(h.get('bm25_sim'))} | {m.get('title')} | {m.get('url')}")
                    else:
                        print(f"  #{i} rr={fmt(h.get('rr_score'))} | vec={fmt(h.get('vec_sim'))} | {m.get('title')} | {m.get('url')}")

                # 分數門檻：若最相關也太低，就視同不檢索
                if not hits or (hits and hits[0].get("rr_score", 0.0) < RERANK_MIN):
                    use_rag = False

            if not use_rag:
                # 純聊天
                messages = [{"role":"user","content": user}]
            else:
                ctx = build_context(hits)
                messages = build_messages(user, ctx)

            print("模型：", end="", flush=True)
            reply = stream_chat(messages, temperature=0.2)
            # 如需多輪上下文可 push 進 history（注意長度）
            # history += [{"role":"user","content":user},{"role":"assistant","content":reply}]
    except KeyboardInterrupt:
        print("\n👋 已中止")

if __name__ == "__main__":
    main()

