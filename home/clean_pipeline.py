# clean_pipeline.py  — 既支援 SQLite，也支援 JSONL 作為輸入
# -*- coding: utf-8 -*-
"""
Pipeline：JSONL/SQLite → 清洗 → 切塊 →（可選）Embedding→Chroma、FTS5

例子：
# 從 data/ 讀 JSONL（支援萬用字元）
python clean_pipeline.py \
  --in-jsonl "data/*.jsonl" \
  --out-dir out --out-people-dir out_people \
  --out-chunk-dir out_chunk --out-people-chunk-dir out_people_chunk \
  --target-chars 1000 --overlap 150 \
  --do-embed --embed-db-path chroma_db --embed-collection campus_news_bgem3 --embed-device cuda \
  --do-fts --fts-db-path fts5_db/fts5.db

# 仍可用舊的 SQLite 模式（不變）
python clean_pipeline.py \
  --db /home/data/site/site.db \
  --table latest_news \
  --out-dir out --out-people-dir out_people \
  --out-chunk-dir out_chunk --out-people-chunk-dir out_people_chunk \
  --target-chars 1000 --overlap 150
"""
import argparse, json, re, hashlib, uuid, unicodedata, sqlite3, glob
from pathlib import Path
from datetime import timezone, timedelta
from dateutil import parser as dtp
import pandas as pd
from bs4 import BeautifulSoup

# ====== 共用工具 ======
TZ_TAIPEI = timezone(timedelta(hours=8))

def nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s) if isinstance(s, str) else s

def strip_html(s: str) -> str:
    if not isinstance(s, str) or not s.strip():
        return ""
    s = re.sub(r"(?i)<br\s*/?>", "\n", s)
    return BeautifulSoup(s, "html.parser").get_text("\n")

def normalize_ws(s: str) -> str:
    if not s: return ""
    s = s.replace("\xa0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

ROC_PAT = re.compile(r"(?P<y>1\d{2})\s*年\s*(?P<m>\d{1,2})\s*月\s*(?P<d>\d{1,2})\s*日")
def convert_roc_dates(text: str) -> str:
    def _repl(m):
        y = 1911 + int(m.group("y")); mth = int(m.group("m")); day = int(m.group("d"))
        return f"{y:04d}-{mth:02d}-{day:02d}"
    return ROC_PAT.sub(_repl, text)

BULLET_PAT = re.compile(r"""^\s*(
    [\u2022\u2023\u25CF\u25A0\u25E6•‧·．\-–—]|
    \(\d+\)|\d+[\.、)]|
    \([一二三四五六七八九十]\)|[一二三四五六七八九十]、|
    \([甲乙丙丁]\)|[甲乙丙丁]、|
    \([A-Za-z]\)|[A-Za-z][\.\)]
)\s*""", re.X)

def normalize_bullets(line: str) -> str:
    return BULLET_PAT.sub("- ", line)

def fix_lines(text: str) -> str:
    lines = [normalize_bullets(nfkc(x)) for x in text.split("\n")]
    return "\n".join(normalize_ws(x) for x in lines if x is not None)

def to_iso8601_tw(s: str) -> str:
    if not s: return None
    try:
        dt = dtp.parse(s)
    except Exception:
        return None
    if not dt.tzinfo:
        dt = dt.replace(tzinfo=TZ_TAIPEI)
    return dt.astimezone(TZ_TAIPEI).isoformat()

def make_checksum(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        if p is None: p = ""
        h.update(str(p).encode("utf-8"))
    return h.hexdigest()

# 將「一上/二下」等轉成「大一上學期/大二下學期」
def _sem_to_zh_grade(sem: str) -> str:
    if not isinstance(sem, str): return ""
    sem = sem.strip()
    mapping = {
        "一上": "大一上學期", "一下": "大一下學期",
        "二上": "大二上學期", "二下": "大二下學期",
        "三上": "大三上學期", "三下": "大三下學期",
        "四上": "大四上學期", "四下": "大四下學期",
    }
    return mapping.get(sem, sem)

def to_natural_sentence_from_obj(obj: dict) -> str:
    """
    把課表/表格列類型物件，轉成對檢索友善的一句自然語句。
    其他型別回空字串（不動原文）。
    """
    if obj.get("type") not in ("course_table_row", "table_row"):
        return ""

    sem   = (obj.get("semester") or "").strip()
    cat   = (obj.get("category") or "").strip()              # 例：基礎共同必修 / 專業必修 / 選修
    sub   = (obj.get("sub_category") or "").strip()          # 例：語文能力 / 程式設計類
    name  = (obj.get("course_name") or "").strip()
    cred  = obj.get("credit")
    title = (obj.get("title") or "").strip()                 # 例：大學部109學年度入學學生必修科目表

    # 學期詞友善化
    sem_zh = _sem_to_zh_grade(sem)

    # 片語組裝（缺哪個欄位就略過）
    parts = []
    if title:
        parts.append(title)
    if sem_zh:
        parts.append(f"在{sem_zh}")
    elif sem:
        parts.append(f"在{sem}")
    if cat:
        parts.append(f"屬於{cat}")
    if sub:
        parts.append(f"{sub}類")
    if name:
        parts.append(f"課程名稱為「{name}」")
    if cred is not None and str(cred) != "":
        parts.append(f"學分數為 {cred} 學分")

    sent = "，".join(parts).strip("，")
    if sent and not sent.endswith("。"):
        sent += "。"
    return sent

# === LLM 自然語句化開關與設定 ===
import os
import requests
USE_LLM_NATURALIZE = os.getenv("USE_LLM_NATURALIZE", "1") != "0"   # 1=啟用 LLM 自然語句化
LLM_NAT_MODEL = os.getenv("LLM_NAT_MODEL", "cwchang/llama-3-taiwan-8b-instruct:latest")
LLM_NAT_URL   = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
LLM_TIMEOUT_S = int(os.getenv("LLM_NAT_TIMEOUT", "30"))
NAT_CACHE_PATH = os.getenv("NATURALIZE_CACHE", "out/naturalize_cache.jsonl")

# 簡單 JSONL 快取（key = checksum 或 (doc_id, chunk-source)）
class NatCache:
    def __init__(self, path: str):
        self.path = Path(path)
        self.map = {}
        if self.path.exists():
            with open(self.path, "r", encoding="utf-8") as fr:
                for line in fr:
                    try:
                        obj = json.loads(line)
                        self.map[obj["key"]] = obj["text"]
                    except Exception:
                        pass

    def get(self, key: str):
        return self.map.get(key)

    def put(self, key: str, text: str):
        self.map[key] = text
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fw:
            fw.write(json.dumps({"key": key, "text": text}, ensure_ascii=False) + "\n")

def llm_naturalize_record(obj: dict) -> str:
    """
    用本地 Ollama 模型，將一筆 JSON 轉成單句自然語句（繁中）。
    嚴格限制：只能使用提供欄位；缺什麼就省略；不得杜撰。
    回傳：自然語句（可能為空字串）
    """
    system = (
        "你是資料轉寫助手。請將輸入的 JSON 欄位，轉寫成一段簡潔的繁體中文單句。\n"
        "規則：\n"
        "1) 只能使用 JSON 中出現的資訊；不得推測或新增內容；數字與名稱需原封不動。\n"
        "2) 缺少的欄位就省略，不要猜測。\n"
        "3) 僅輸出一個句子，不要列表、不要額外解釋、不要加引號。\n"
        "4) 語氣中性、正式。\n"
    )
    # 你可依需求調整關鍵欄位的呈現順序
    user = (
        "將下列 JSON 轉為單句描述，優先包含：學期(semester)、類別(category / sub_category)、"
        "課程名稱(course_name)、學分(credit)，再包含標題(title)。若沒有就略過：\n"
        + json.dumps(obj, ensure_ascii=False)
    )

    payload = {
        "model": LLM_NAT_MODEL,
        "messages": [{"role":"system","content":system},{"role":"user","content":user}],
        "stream": False,
        "options": {"temperature": 0.1}
    }
    try:
        r = requests.post(LLM_NAT_URL, json=payload, timeout=LLM_TIMEOUT_S)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and "message" in data and "content" in data["message"]:
            text = (data["message"]["content"] or "").strip()
        else:
            text = (data.get("content") or "").strip() if isinstance(data, dict) else ""
        # 簡單清理
        text = text.splitlines()[0].strip("「」\"' ").strip()
        if text and not text.endswith("。"):
            text += "。"
        return text
    except Exception:
        return ""
    
def validate_naturalized(obj: dict, sent: str) -> bool:
    """
    確認 LLM 輸出沒有遺失關鍵欄位值（若存在就應被包含），避免幻覺/漏資訊。
    只做子字串檢查（保守），不過度嚴格以免常常退回。
    """
    if not sent: return False
    ok = True
    for key in ["semester", "category", "sub_category", "course_name"]:
        v = obj.get(key)
        if isinstance(v, str) and v.strip():
            if v.strip() not in sent:
                # 學期可接受「一上→大一上學期」這種別名，因此只要求其中之一命中
                if key == "semester":
                    alt = _sem_to_zh_grade(v)
                    if alt and alt not in sent:
                        ok = False
                else:
                    ok = False
    # 數字學分：要求出現原始數字
    cred = obj.get("credit", None)
    if cred is not None and str(cred) not in sent:
        ok = False
    return ok

# ====== SQLite → DataFrame（沿用舊版）======
def import_sqlite_to_df(db_path: str, table: str = "latest_news") -> pd.DataFrame:
    con = sqlite3.connect(str(Path(db_path)))
    try:
        df = pd.read_sql(f'SELECT * FROM "{table}"', con)
    finally:
        con.close()

    df = df.rename(columns={
        "標題": "title",
        "連結": "url",
        "發布時間": "published_at",
        "內文": "html_text",
        "pdf內容": "pdf_text",
    })
    for k in ["title", "url", "published_at", "html_text", "pdf_text"]:
        if k not in df.columns:
            df[k] = ""
    return df

# ====== JSONL → DataFrame（新增）======
def import_jsonl_to_df(in_jsonl_glob: str) -> pd.DataFrame:
    """
    從 JSONL 檔（可萬用字元）讀入，統一轉成欄位：
    title, url, published_at, text, html_text, pdf_text
    - 若來源已是 cleaned（有 text），保留並輕量清洗
    - 若只有 html_text / pdf_text，合併為 text
    - 若是「表格列」類型（course_table_row），就把 text = row_text 或組合欄位
    """
    paths = sorted({str(p) for g in in_jsonl_glob.split(",") for p in Path().glob(g.strip())})
    rows = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as fr:
            for line in fr:
                line = line.strip()
                if not line: continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                # 常見鍵名取值（存在則帶過來）
                title = obj.get("title") or obj.get("標題") or ""
                url = obj.get("url") or obj.get("連結") or obj.get("pdf_url") or ""
                published_at = obj.get("published_at") or obj.get("發布時間") or ""
                html_text = obj.get("html_text") or obj.get("source_html_text") or obj.get("內文") or ""
                pdf_text = obj.get("pdf_text") or obj.get("source_pdf_text") or obj.get("pdf內容") or ""

                text = obj.get("text") or ""
                # 若是課表列（你現在的 required_courses.cleaned.jsonl）
                if not text and obj.get("type") in ("course_table_row", "table_row"):
                    # 優先 row_text；沒有就組合欄位
                    rt = obj.get("row_text") or ""
                    if not rt:
                        parts = [
                            obj.get("semester",""),
                            obj.get("category",""),
                            obj.get("sub_category",""),
                            obj.get("course_name",""),
                            f'{obj.get("credit","")} 學分' if obj.get("credit") is not None else ""
                        ]
                        rt = "｜".join([str(x) for x in parts if str(x)])
                    text = rt

                # 若 text 仍為空，退回合併 html_text/pdf_text
                if not text:
                    text = ((html_text or "").strip() + "\n\n" + (pdf_text or "").strip()).strip()

                # ====== 新增：以 LLM 產生繁中自然語句（有快取與回退） ======
                nl_prefix = ""
                NAT_TYPES = {"course_table_row", "table_row", "department_member_row"}
                if USE_LLM_NATURALIZE and obj.get("type") in NAT_TYPES:
                    cache = getattr(import_jsonl_to_df, "_nat_cache", None)
                    if cache is None:
                        cache = NatCache(NAT_CACHE_PATH)
                        setattr(import_jsonl_to_df, "_nat_cache", cache)

                    cache_key = obj.get("checksum") or obj.get("checksum_clean") or make_checksum(
                        obj.get("title"), obj.get("url"), obj.get("published_at"), json.dumps(obj, ensure_ascii=False)
                    )
                    cached = cache.get(cache_key)
                    if cached:
                        nl_prefix = cached
                    else:
                        # 1) 先嘗試 LLM
                        cand = llm_naturalize_record(obj)
                        if cand and validate_naturalized(obj, cand):
                            nl_prefix = cand
                            cache.put(cache_key, nl_prefix)
                        else:
                            # 2) 回退：用既有的 rule-based 句子
                            rb = to_natural_sentence_from_obj(obj)  # 你現有的函式
                            if rb:
                                nl_prefix = rb
                                cache.put(cache_key, nl_prefix)

                # 把自然語句前綴在 text 前面
                if nl_prefix:
                    text = f"{nl_prefix}\n{text}" if text else nl_prefix
                # ==============================================
                # doc_id / checksum：沿用既有鍵，沒有就現算
                doc_id = obj.get("doc_id") or obj.get("checksum") or None
                checksum = obj.get("checksum") or obj.get("checksum_clean") or None
                if not (doc_id and checksum):
                    checksum = make_checksum(title, url, published_at, text)
                    if not doc_id:
                        doc_id = checksum

                rows.append({
                    "doc_id": doc_id,
                    "title": title,
                    "url": url,
                    "published_at": published_at,
                    "html_text": html_text,
                    "pdf_text": pdf_text,
                    "text": text,
                    "checksum": checksum,
                })

    if not rows:
        return pd.DataFrame(columns=["doc_id","title","url","published_at","html_text","pdf_text","text","checksum"])
    df = pd.DataFrame(rows)
    # 只以 checksum 去重；不要再用 doc_id 去重（因為課表每列共用同一 doc_id）
    df = df.drop_duplicates(subset=["checksum"])
    return df

# ====== 清洗 → DataFrame（沿用）======
def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df["html_text"] = df["html_text"].fillna("").map(strip_html).map(nfkc).map(convert_roc_dates).map(fix_lines)
    df["pdf_text"]  = df["pdf_text"].fillna("").map(nfkc).map(convert_roc_dates).map(fix_lines)
    # 若 text 已有（例如課表列），仍做輕量正規化；否則用 html+pdf 合併
    has_text = df["text"].fillna("").map(bool)
    df.loc[has_text, "text"] = df.loc[has_text, "text"].map(nfkc).map(convert_roc_dates).map(fix_lines)
    df.loc[~has_text, "text"] = (df.loc[~has_text, "html_text"].str.strip() + "\n\n" + df.loc[~has_text, "pdf_text"].str.strip()).str.strip()
    df["text"] = df["text"].map(normalize_ws)
    df["published_at"] = df["published_at"].map(to_iso8601_tw)
    df["url"] = df["url"].map(lambda s: s.strip() if isinstance(s, str) else s)
    # 若 doc_id 不穩，重新計算 checksum；doc_id 仍保留原值
    df["checksum"] = [make_checksum(t, u, p, x) for t, u, p, x in zip(df["title"], df["url"], df["published_at"], df["text"])]
    df = df.drop_duplicates(subset=["checksum"])
    return df

# ====== 清洗 → JSONL（沿用）======
def write_clean_outputs(df: pd.DataFrame, out_dir: Path, out_people_dir: Path, base_name: str = "clean_latest_news") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_people_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{base_name}.clean.jsonl"
    pretty_file = out_people_dir / f"{base_name}.clean.pretty.json"
    out_file.unlink(missing_ok=True)
    with open(out_file, "w", encoding="utf-8") as fw_jsonl, \
         open(pretty_file, "w", encoding="utf-8") as fw_pretty:
        for row in df.to_dict(orient="records"):
            rec = {
                "doc_id": row.get("doc_id") or row["checksum"],
                "title": row["title"],
                "url": row["url"],
                "published_at": row["published_at"],
                "text": row["text"],
                "source_html_text": row["html_text"],
                "source_pdf_text": row["pdf_text"],
                "checksum": row["checksum"],
            }
            fw_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fw_pretty.write(json.dumps(rec, ensure_ascii=False, indent=2) + "\n\n")
    print(f"[OK] 清洗完成 → {out_file}")
    print(f"[OK] 清洗完成 → {pretty_file}")
    return out_file

# ====== 切塊（沿用）======
PUNCTS = "。！？!?；;"

def split_sentences(text: str):
    text = re.sub(rf"([{PUNCTS}])", r"\1<S>", text.replace("\r\n","\n").replace("\r","\n"))
    text = text.replace("\n", "<S>")
    parts = [p.strip() for p in text.split("<S>")]
    return [p for p in parts if p]

def hard_slice_with_overlap(text: str, target: int, overlap: int):
    i, n = 0, len(text); out=[]
    while i < n:
        end = min(n, i + target)
        out.append(text[i:end])
        if end >= n: break
        i = end - overlap if end - overlap > i else end
    return out

def _normalize_for_hash(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def make_chunk_id(doc_id: str, text: str) -> str:
    h = hashlib.sha1()
    h.update((doc_id or "").encode("utf-8"))
    h.update(b"\n")
    h.update(_normalize_for_hash(text).encode("utf-8"))
    return h.hexdigest()

def chunk_text(text: str, target_chars: int, overlap_chars: int):
    sents = split_sentences(text)
    if not sents: return []
    chunks=[]; cur=""
    def flush():
        nonlocal cur
        t = cur.strip()
        if t: chunks.append(t)
        cur=""
    for s in sents:
        if len(s) > target_chars:
            flush()
            chunks += [seg.strip() for seg in hard_slice_with_overlap(s, target_chars, overlap_chars)]
            continue
        if not cur:
            cur = s
        elif len(cur) + 1 + len(s) <= target_chars:
            cur = f"{cur}\n{s}"
        else:
            t = cur.strip()
            if t:
                tail = t[-overlap_chars:] if len(t) > overlap_chars else t
                chunks.append(t)
                cur = f"{tail}\n{s}"
            else:
                cur = s
    flush()
    return chunks

def chunk_file(clean_jsonl: Path, out_chunk_dir: Path, out_people_chunk_dir: Path,
               target_chars=1000, overlap=150, base_name: str = "clean_latest_news") -> Path:
    out_chunk_dir.mkdir(parents=True, exist_ok=True)
    out_people_chunk_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_chunk_dir / f"{base_name}.chunks.jsonl"
    pretty_path = out_people_chunk_dir / f"{base_name}.chunks.pretty.json"
    out_path.unlink(missing_ok=True)

    total_docs = 0
    total_chunks = 0

    with open(clean_jsonl, "r", encoding="utf-8") as fr, \
         open(out_path, "w", encoding="utf-8") as fw_jsonl, \
         open(pretty_path, "w", encoding="utf-8") as fw_pretty:

        for line in fr:
            if not line.strip():
                continue
            rec = json.loads(line)
            total_docs += 1

            doc_id = rec.get("doc_id")
            text = (rec.get("text") or "").replace("\r\n", "\n").replace("\r", "\n").strip()

            if not text:
                out = {
                    "doc_id": doc_id,
                    "chunk_id": make_chunk_id(doc_id, ""),
                    "type": "text",
                    "text": "",
                    "meta": {
                        "title": rec.get("title"),
                        "url": rec.get("url"),
                        "published_at": rec.get("published_at"),
                        "checksum": rec.get("checksum"),
                        "chunk_index": 0,
                    },
                }
                fw_jsonl.write(json.dumps(out, ensure_ascii=False) + "\n")
                fw_pretty.write(json.dumps(out, ensure_ascii=False, indent=2) + "\n\n")
                total_chunks += 1
                continue

            parts = chunk_text(text, target_chars, overlap)
            for idx, part in enumerate(parts):
                part = part.strip()
                out = {
                    "doc_id": doc_id,
                    "chunk_id": make_chunk_id(doc_id, part),
                    "type": "text",
                    "text": part,
                    "meta": {
                        "title": rec.get("title"),
                        "url": rec.get("url"),
                        "published_at": rec.get("published_at"),
                        "checksum": rec.get("checksum"),
                        "chunk_index": idx,
                    },
                }
                fw_jsonl.write(json.dumps(out, ensure_ascii=False) + "\n")
                fw_pretty.write(json.dumps(out, ensure_ascii=False, indent=2) + "\n\n")
                total_chunks += 1

    print(f"[OK] 完成切塊 → {out_path}")
    print(f"[OK] 完成切塊 → {pretty_path}")
    print(f"文件數: {total_docs}；切塊數: {total_chunks}")
    return out_path

# ====== Embedding → Chroma（沿用）======
def embed_and_index_chroma(chunks_jsonl_path: Path, chroma_dir: Path, collection_name: str,
                           device: str = "cuda", batch_size: int = 64,
                           site_slug: str = None):
    chroma_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import chromadb
    from FlagEmbedding import BGEM3FlagModel

    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_or_create_collection(name=collection_name)
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device=device)

    def l2_normalize(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        return v / n

    ids, docs, metas = [], [], []
    with open(chunks_jsonl_path, "r", encoding="utf-8") as fr:
        for line in fr:
            if not line.strip(): continue
            rec = json.loads(line)
            cid = rec.get("chunk_id"); text = rec.get("text") or ""
            meta = {**(rec.get("meta") or {}), "type": rec.get("type", "text")}
            if site_slug:
                meta["site_slug"] = site_slug
            if not cid or not text.strip(): continue
            ids.append(cid); docs.append(text); metas.append(meta)

    total = 0
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i:i+batch_size]
        batch_ids  = ids[i:i+batch_size]
        batch_meta = metas[i:i+batch_size]

        out = model.encode(batch_docs, batch_size=len(batch_docs),
                           return_dense=True, return_sparse=False, return_colbert_vecs=False)
        import numpy as np
        vecs = l2_normalize(np.array(out["dense_vecs"], dtype=np.float32)).tolist()

        if hasattr(collection, "upsert"):
            collection.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_meta, embeddings=vecs)
        else:
            try:
                collection.add(ids=batch_ids, documents=batch_docs, metadatas=batch_meta, embeddings=vecs)
            except Exception:
                try: collection.delete(ids=batch_ids)
                except Exception: pass
                collection.add(ids=batch_ids, documents=batch_docs, metadatas=batch_meta, embeddings=vecs)

        total += len(batch_docs)
        print(f"  - 已寫入 {total} 筆 ...")
    print(f"✅ Chroma 索引完成：{total} 筆 → {chroma_dir}（collection='{collection_name}'）")

# ====== 建立/更新 FTS5（沿用）======
def build_or_update_fts5(chunks_jsonl_path: Path, fts_db_path: Path, site_slug: str = None):
    fts_db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(fts_db_path))
    cur = con.cursor()
    cur.execute("""
      CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
      USING fts5(
        title, text, url, published_at, doc_id, chunk_id, type, site_slug,
        tokenize='unicode61'
      );
    """)
    con.commit()

    inserted = 0
    with open(chunks_jsonl_path, "r", encoding="utf-8") as fr:
        for line in fr:
            if not line.strip(): continue
            rec = json.loads(line); meta = rec.get("meta") or {}; cid = rec.get("chunk_id")
            if not cid: continue
            vals = (
                meta.get("title") or "",
                rec.get("text") or "",
                meta.get("url") or "",
                meta.get("published_at") or "",
                rec.get("doc_id") or "",
                cid,
                rec.get("type") or "text",
                site_slug or "",
            )
            cur.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (cid,))
            cur.execute("""INSERT INTO chunks_fts (title, text, url, published_at, doc_id, chunk_id, type, site_slug)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", vals)
            inserted += 1
            if inserted % 1000 == 0:
                con.commit()
    con.commit(); con.close()
    print(f"✅ FTS5 索引更新完成：{inserted} 筆 → {fts_db_path}")

# ====== 刪除工具（沿用）======
def reset_chroma(chroma_dir: Path, collection_name: str):
    import chromadb
    chroma_dir.mkdir(parents=True, exist_ok=True)
    try:
        client = chromadb.PersistentClient(path=str(chroma_dir))
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
    except Exception:
        pass

def reset_fts5(fts_db_path: Path):
    fts_db_path.parent.mkdir(parents=True, exist_ok=True)
    fts_db_path.unlink(missing_ok=True)

# ====== 主流程：新增 JSONL 入口 ======
def run_pipeline_jsonl(in_jsonl_glob: str,
                       out_dir="out", out_people_dir="out_people",
                       out_chunk_dir="out_chunk", out_people_chunk_dir="out_people_chunk",
                       target_chars=1000, overlap=150,
                       do_embed: bool = False,
                       embed_db_path: str = "chroma_db",
                       embed_collection: str = "campus_news_bgem3",
                       embed_device: str = "cuda",
                       do_fts: bool = False,
                       fts_db_path: str = "fts5_db/fts5.db",
                       site_slug: str = None,
                       rebuild_chroma: bool = False,
                       rebuild_fts: bool = False):
    base = Path(".").resolve()
    out_dir_p = (base / out_dir).resolve()
    out_people_dir_p = (base / out_people_dir).resolve()
    out_chunk_dir_p = (base / out_chunk_dir).resolve()
    out_people_chunk_dir_p = (base / out_people_chunk_dir).resolve()

    df = import_jsonl_to_df(in_jsonl_glob)
    df = clean_df(df)
    
    # 以 site_slug 優先；沒有就用輸入檔名（去副檔名）
    # 注意：in_jsonl_glob 在這裡會是單一檔路徑
    base_name = (site_slug or Path(in_jsonl_glob).stem)

    clean_jsonl = write_clean_outputs(df, out_dir_p, out_people_dir_p, base_name=base_name)
    chunks_jsonl = chunk_file(clean_jsonl, out_chunk_dir_p, out_people_chunk_dir_p,
                              target_chars=target_chars, overlap=overlap, base_name=base_name)

    if do_embed:
        if rebuild_chroma:
            reset_chroma(Path(embed_db_path), embed_collection)
        embed_and_index_chroma(
            chunks_jsonl_path=Path(chunks_jsonl),
            chroma_dir=Path(embed_db_path),
            collection_name=embed_collection,
            device=embed_device,
            batch_size=64,
            site_slug=site_slug,
        )
    if do_fts:
        if rebuild_fts:
            reset_fts5(Path(fts_db_path))
        build_or_update_fts5(
            chunks_jsonl_path=Path(chunks_jsonl),
            fts_db_path=Path(fts_db_path),
            site_slug=site_slug,
        )

    return {
        "base_name": base_name,
        "clean_jsonl": str(clean_jsonl),
        "chunks_jsonl": str(chunks_jsonl),
        "out_dir": str(out_dir_p),
        "out_chunk_dir": str(out_chunk_dir_p),
        "chroma_db": str(Path(embed_db_path).resolve()) if do_embed else None,
        "fts_db": str(Path(fts_db_path).resolve()) if do_fts else None,
    }

# ====== 主流程：沿用 SQLite 入口 ======
def run_pipeline_sqlite(db_path: str,
                        table: str = "latest_news",
                        out_dir="out", out_people_dir="out_people",
                        out_chunk_dir="out_chunk", out_people_chunk_dir="out_people_chunk",
                        target_chars=1000, overlap=150,
                        do_embed: bool = False,
                        embed_db_path: str = "chroma_db",
                        embed_collection: str = "campus_news_bgem3",
                        embed_device: str = "cuda",
                        do_fts: bool = False,
                        fts_db_path: str = "fts5_db/fts5.db",
                        site_slug: str = None,
                        rebuild_chroma: bool = False,
                        rebuild_fts: bool = False):
    base = Path(db_path).resolve().parent
    out_dir_p = (base / out_dir).resolve()
    out_people_dir_p = (base / out_people_dir).resolve()
    out_chunk_dir_p = (base / out_chunk_dir).resolve()
    out_people_chunk_dir_p = (base / out_people_chunk_dir).resolve()

    df = import_sqlite_to_df(db_path, table=table)
    df = clean_df(df)
    clean_jsonl = write_clean_outputs(df, out_dir_p, out_people_dir_p)
    chunks_jsonl = chunk_file(clean_jsonl, out_chunk_dir_p, out_people_chunk_dir_p,
                              target_chars=target_chars, overlap=overlap)

    if do_embed:
        if rebuild_chroma:
            reset_chroma(Path(embed_db_path), embed_collection)
        embed_and_index_chroma(
            chunks_jsonl_path=Path(chunks_jsonl),
            chroma_dir=Path(embed_db_path),
            collection_name=embed_collection,
            device=embed_device,
            batch_size=64,
            site_slug=site_slug,
        )
    if do_fts:
        if rebuild_fts:
            reset_fts5(Path(fts_db_path))
        build_or_update_fts5(
            chunks_jsonl_path=Path(chunks_jsonl),
            fts_db_path=Path(fts_db_path),
            site_slug=site_slug,
        )

    return {
        "clean_jsonl": str(clean_jsonl),
        "chunks_jsonl": str(chunks_jsonl),
        "out_dir": str(out_dir_p),
        "out_chunk_dir": str(out_chunk_dir_p),
        "chroma_db": str(Path(embed_db_path).resolve()) if do_embed else None,
        "fts_db": str(Path(fts_db_path).resolve()) if do_fts else None,
    }

# ====== CLI ======
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-jsonl", default="data/*.jsonl",
                    help='JSONL 輸入路徑，支援萬用字元（例如："data/*.jsonl,data/cse/*.jsonl"）')
    ap.add_argument("--db", default=None, help="（舊）SQLite DB 路徑，例如 /home/data/site/site.db")
    ap.add_argument("--table", default="latest_news", help="（舊）資料表名稱（SQLite 模式）")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--out-people-dir", default="out_people")
    ap.add_argument("--out-chunk-dir", default="out_chunk")
    ap.add_argument("--out-people-chunk-dir", default="out_people_chunk")
    ap.add_argument("--target-chars", type=int, default=1000)
    ap.add_argument("--overlap", type=int, default=150)
        # ==== 新：把 do-embed / do-fts 預設改為 True，並提供關閉開關 ====
    embed_group = ap.add_mutually_exclusive_group()
    embed_group.add_argument("--do-embed", dest="do_embed", action="store_true",
                             help="建立/更新 Chroma 向量庫（預設：開）")
    embed_group.add_argument("--no-embed", dest="do_embed", action="store_false",
                             help="不要建立 Chroma 向量庫")
    ap.set_defaults(do_embed=True)

    fts_group = ap.add_mutually_exclusive_group()
    fts_group.add_argument("--do-fts", dest="do_fts", action="store_true",
                           help="建立/更新 FTS5 BM25 索引（預設：開）")
    fts_group.add_argument("--no-fts", dest="do_fts", action="store_false",
                           help="不要建立 FTS5 索引")
    ap.set_defaults(do_fts=True)
    # ============================================================
    ap.add_argument("--embed-db-path", default="chroma_db")
    ap.add_argument("--embed-collection", default="campus_news_bgem3")
    ap.add_argument("--embed-device", default="cuda")
    ap.add_argument("--fts-db-path", default="fts5_db/fts5.db")
    ap.add_argument("--site-slug", default=None, help="若未指定，預設用輸入檔案名稱")
    ap.add_argument("--rebuild-chroma", action="store_true")
    ap.add_argument("--rebuild-fts", action="store_true")
    args = ap.parse_args()

    import glob
    import json as _json

    # args.in_jsonl = "data/*.jsonl"
    for pattern in args.in_jsonl.split(","):
        for file_path in glob.glob(pattern.strip()):
            infile = Path(file_path)
            current_filename = infile.stem  # 不含副檔名
            # 或者 infile.name  # 含副檔名
            print("正在處理：", infile)
            print("存入變數 current_filename =", current_filename)



    if args.in_jsonl:
       # 1) 展開萬用字元，得到每個實際檔案
        files = []
        for pattern in args.in_jsonl.split(","):
            files.extend(glob.glob(pattern.strip()))
        files = sorted({str(Path(p)) for p in files})
        if not files:
            raise SystemExit(f"找不到任何檔案：{args.in_jsonl}")

        results = []
        # 2) 逐檔呼叫一次 pipeline，每檔用各自的 site_slug
        for fp in files:
            infile = Path(fp)
            current_slug = args.site_slug or infile.stem
            print(f"▶ 處理檔案：{infile} （site_slug={current_slug}）")

            res = run_pipeline_jsonl(
                in_jsonl_glob=str(infile),              # ← 只處理這一個檔
                out_dir=args.out_dir,
                out_people_dir=args.out_people_dir,
                out_chunk_dir=args.out_chunk_dir,
                out_people_chunk_dir=args.out_people_chunk_dir,
                target_chars=args.target_chars,
                overlap=args.overlap,
                do_embed=args.do_embed,
                embed_db_path=args.embed_db_path,
                embed_collection=args.embed_collection,
                embed_device=args.embed_device,
                do_fts=args.do_fts,
                fts_db_path=args.fts_db_path,
                site_slug=current_slug,                 # ← 這一檔的 site_slug
                rebuild_chroma=args.rebuild_chroma,
                rebuild_fts=args.rebuild_fts,
            )
            results.append({"file": str(infile), "site_slug": current_slug, **res})

        print(_json.dumps(results, ensure_ascii=False, indent=2))
        return
    elif args.db:
        res = run_pipeline_sqlite(
            db_path=args.db,
            table=args.table,
            out_dir=args.out_dir,
            out_people_dir=args.out_people_dir,
            out_chunk_dir=args.out_chunk_dir,
            out_people_chunk_dir=args.out_people_chunk_dir,
            target_chars=args.target_chars,
            overlap=args.overlap,
            do_embed=args.do_embed,
            embed_db_path=args.embed_db_path,
            embed_collection=args.embed_collection,
            embed_device=args.embed_device,
            do_fts=args.do_fts,
            fts_db_path=args.fts_db_path,
            site_slug=args.site_slug or "cse_latest_news",
            rebuild_chroma=args.rebuild_chroma,
            rebuild_fts=args.rebuild_fts,
        )
    else:
        raise SystemExit("請指定 --in-jsonl 或 --db 其中之一。")

    print(json.dumps(res, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
