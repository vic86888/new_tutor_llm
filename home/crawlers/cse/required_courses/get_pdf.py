# get_pdf.py  (refactor: 不再寫檔，只回傳資料)
# -*- coding: utf-8 -*-
import re, io, requests
import pdfplumber
import json

# ---- 文字規則（原樣保留）----
TERMS = ['一上','一下','二上','二下','三上','三下','四上','四下']
RE_CREDIT_PARENS = re.compile(r"[（(]\s*(\d+)\s*[）)]")       # (3) 或（3）
RE_CREDIT_WORD   = re.compile(r"(\d+)\s*學分")               # 3學分
RE_NOTES_CUE     = re.compile(r"必須達|方可修|先修|限修")
STRUCT_TOKENS    = {"基礎","共同","必修","專業","系訂","小計","學分小計","學分","系訂專業必修","基礎共同必修","公民","素養"}

def _N(s: str) -> str:
    return re.sub(r"\s+"," ", (s or "").replace("\u3000"," ").replace("\xa0"," ").strip())

def _subcat(text: str) -> str:
    t = _N(text)
    if "勞作教育" in t or "公民素養" in t: return "公民素養"
    if "社會設計" in t:                    return "公民素養"
    if "體育" in t:                        return "體育"
    if any(k in t for k in ["國語文","能力表達","英文","日文"]) or ("語文" in t and "能力" in t):
        return "語文能力"
    return ""

def _major_from_sub(sub: str) -> str:
    return "基礎共同必修" if sub in ("語文能力","體育","公民素養") else "系訂專業必修"

def _clean_name(raw_name: str, block_text: str) -> str:
    s = _N(raw_name)
    s = _N(" ".join([w for w in s.split() if w not in STRUCT_TOKENS]))
    m = re.fullmatch(r"語文\s+(英文[一二三四])\s+能力", s)
    if m: s = m.group(1)
    if "體育" in s and ("必修" in raw_name or s.count("體育") >= 2):
        s = "體育"
    if "勞作教育" in s:
        s = s.replace("Ａ","A").replace("Ｂ","B")
        m2 = re.search(r"(勞作教育\s+[AB]\s*班)", s)
        s = m2.group(1) if m2 else "勞作教育 班"
    s = s.replace("系訂", "").strip()
    if "社會設計" in raw_name:
        s = "社會設計"
    return s

def extract_text_from_pdf(url: str, out_jsonl_path: str = "courses.jsonl"):
    """
    下載並解析 PDF（『大學部學生必修科目表』），不落地 JSON 檔，直接回傳 dict：
    { "courses": [...], "notes": [...], "count": N }
    解析規則完全沿用原本版本。
    """
    pdf_bytes = requests.get(url, timeout=30).content
    courses, notes = [], []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for pidx, page in enumerate(pdf.pages, start=1):
            words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
            headers = [w for w in words if w["text"] in TERMS]
            if not headers:
                continue
            headers = sorted(headers, key=lambda w: w["x0"])
            centers = [ (w["x0"]+w["x1"])/2 for w in headers ]
            header_y = max(w["bottom"] for w in headers)

            def nearest_col(xmid: float) -> int:
                diffs = [abs(xmid-cx) for cx in centers]
                return int(min(range(8), key=lambda i: diffs[i]))

            col_words = {i: [] for i in range(8)}
            for w in words:
                if w["top"] < header_y - 2:
                    continue
                xmid = (w["x0"]+w["x1"])/2
                col_words[nearest_col(xmid)].append(w)

            blocks = []
            for ci in range(8):
                ws = sorted(col_words[ci], key=lambda w: (w["top"], w["x0"]))
                if not ws:
                    continue
                lines, y_bins = [], []
                for w in ws:
                    y = w["top"]; hit = None
                    for i, yb in enumerate(y_bins):
                        if abs(y - yb) <= 2.5:
                            hit = i; break
                    if hit is None:
                        y_bins.append(y); lines.append([w])
                    else:
                        lines[hit].append(w)
                        y_bins[hit] = (y_bins[hit]*(len(lines[hit])-1) + y) / len(lines[hit])
                pairs = sorted(zip(y_bins, lines), key=lambda t: t[0])
                gaps = [pairs[i+1][0] - pairs[i][0] for i in range(len(pairs)-1)]
                gap_med = sorted(gaps)[len(gaps)//2] if gaps else 18.0
                GAP = max(14.0, min(28.0, gap_med * 1.8))

                cur, last_y = {"col": ci, "top": None, "texts": []}, None
                for y, wsline in pairs:
                    line_txt = _N(" ".join([ww["text"] for ww in sorted(wsline, key=lambda a:a["x0"])]))
                    if not line_txt:
                        continue
                    if cur["top"] is None or (y - last_y) <= GAP:
                        cur["top"] = y if cur["top"] is None else cur["top"]
                        cur["texts"].append(line_txt)
                    else:
                        blocks.append(cur)
                        cur = {"col": ci, "top": y, "texts": [line_txt]}
                    last_y = y
                if cur["texts"]:
                    blocks.append(cur)

            merged = _N(" ".join(w["text"] for w in words))
            seg = merged[merged.find("備註"):] if "備註" in merged else merged
            for piece in re.split(r"\s*(?=(?:\d+[.．]))", seg):
                s = _N(piece)
                if s and RE_NOTES_CUE.search(s):
                    if not s.endswith("。"): s += "。"
                    if s not in notes: notes.append(s)
            if not notes:
                for s in re.split(r"[。；;]\s*", merged):
                    s = _N(s)
                    if RE_NOTES_CUE.search(s):
                        notes.append(s+"。")

            for b in blocks:
                block_text = " ".join(b["texts"])
                tokens = []
                for line in b["texts"]:
                    parts = [p for p in re.split(r"(\([0-9]+\)|（[0-9]+）)", line) if _N(p)]
                    for part in parts:
                        tokens.extend(_N(part).split())

                name_parts = []
                for tok in tokens:
                    m1 = RE_CREDIT_PARENS.fullmatch(tok)
                    m2 = RE_CREDIT_WORD.fullmatch(tok)
                    if m1 or m2:
                        credit = int(m1.group(1)) if m1 else int(m2.group(1))
                        raw_name = _N(" ".join(name_parts)).strip(" -、．.：:")
                        name = _clean_name(raw_name, block_text)
                        if name:
                            sub = _subcat(name) or _subcat(block_text)
                            if "社會設計" in name: sub = "公民素養"
                            if "勞作教育" in name: sub = "公民素養"
                            major = _major_from_sub(sub)
                            courses.append({
                                "來源PDF": url,
                                "頁次": pidx,
                                "學期": TERMS[b["col"]],
                                "課程名稱": name,
                                "學分": credit,
                                "類別大項": major,
                                "子類別": sub
                            })
                        name_parts = []
                    else:
                        name_parts.append(tok)

    # 5) 去重（保留第一筆）
    seen, uniq = set(), []
    for r in courses:
        key = (r["來源PDF"], r["學期"], r["課程名稱"], r["學分"], r["類別大項"], r["子類別"])
        if key in seen: 
            continue
        seen.add(key); uniq.append(r)

    # --- 輸出 JSONL（只有 out_jsonl_path 有給時才寫）---
    if out_jsonl_path:
        with open(out_jsonl_path, "w", encoding="utf-8") as f:
            for course in uniq:
                f.write(json.dumps(course, ensure_ascii=False) + "\n")
            for note in notes:
                note_obj = {"來源PDF": url, "備註": note}
                f.write(json.dumps(note_obj, ensure_ascii=False) + "\n")

    return {"courses": uniq, "notes": notes, "count": len(uniq)}
