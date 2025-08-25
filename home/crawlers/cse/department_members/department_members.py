import requests
from bs4 import BeautifulSoup
import os
import json
import argparse
import hashlib
import datetime
import re

SITE_SLUG = "department_members"
TITLE     = "系所成員名錄"
LANG      = "zh-Hant"

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

def now_iso_tpe() -> str:
    return datetime.datetime.now().astimezone().isoformat()

def sha1_of(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()

def ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

def append_jsonl(record: dict, out_path: str) -> None:
    ensure_dir(out_path)
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def extract_email(s: str) -> str:
    if not s: return ""
    m = EMAIL_RE.search(s)
    return m.group(0) if m else s  # 找不到就原樣保留（和原本行為一致）

# NEW: 載入既有 JSONL 的 checksum_clean（若沒有則回傳空集合）
def load_existing_checksums(path: str) -> set:
    checksums = set()
    if not os.path.exists(path):
        return checksums
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    cs = obj.get("checksum_clean")
                    # 若舊資料沒有 checksum_clean，就退而用 text 算一次
                    if not cs and "text" in obj:
                        cs = sha1_of(obj.get("text", ""))
                    if cs:
                        checksums.add(cs)
                except Exception:
                    # 單行壞掉就跳過，不影響其他行
                    continue
    except Exception:
        pass
    return checksums
# END NEW

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=os.path.join("data", "department_members.jsonl"),
        help="輸出 JSONL 檔案路徑（預設：data/department_members.jsonl）"
    )
    args = parser.parse_args()

    # NEW: 先讀取既有 checksum，避免重複寫入
    existing_checksums = load_existing_checksums(args.out)

    listing_url = "https://cse.ttu.edu.tw/p/412-1058-157.php?Lang=zh-tw"
    res = requests.get(listing_url)
    soup = BeautifulSoup(res.text, 'html.parser')

    for mdetail in soup.find_all(class_="mdetail"):
        all_info = []

        # 人名（藍字）
        name_nodes = mdetail.find_all(style="color:#0000cd;")
        name = ', '.join([x.get_text(strip=True) for x in name_nodes if x.get_text(strip=True)])
        if name:
            all_info.append(name)

        # 其他欄位
        info = mdetail.find_all('p')
        for add in info:
            add_str = add.get_text(strip=True)
            trash = False
            for x in all_info:
                if add_str in x or add_str == '學術研究發表｜研究計畫':
                    trash = True
                    break
            if not trash:
                all_info.append(add_str)

        # 兼任分支（維持既有規則）
        if all_info and '兼任' in all_info[0]:
            person   = all_info[0]
            email    = all_info[4] if len(all_info) > 4 else ""
            email    = extract_email(email)
            metadata = "。".join(all_info[1:4])
            phone    = ""
            office   = ""
        else:
            if not all_info:
                continue
            person   = all_info[0] if len(all_info) > 0 else ""
            phone    = all_info[1] if len(all_info) > 1 else ""
            email    = extract_email(all_info[2] if len(all_info) > 2 else "")
            office   = all_info[3] if len(all_info) > 3 else ""
            metadata = "".join(all_info[4:]) if len(all_info) > 4 else ""

        # —— 與向量庫一致的欄位 —— #
        # 同一人穩定 doc_id：避免重複寫入
        doc_id = sha1_of(f"{SITE_SLUG}|{person}")

        # 文字欄（和課表 row 一樣用全形直線分隔）
        # 格式：人物｜電話｜信箱｜辦公室｜（metadata）
        row_text = "｜".join([x for x in [person, phone, email, office, metadata] if str(x).strip()])

        # NEW: 先算 checksum，若已存在則跳過
        checksum = sha1_of(row_text)
        if checksum in existing_checksums:
            continue
        existing_checksums.add(checksum)
        # END NEW

        record = {
            "site_slug": SITE_SLUG,
            "url": listing_url,
            "pdf_url": None,                     # 與課表欄位對齊，這邊沒有就給 None
            "title": TITLE,
            "source_kind": "html",
            "type": "department_member_row",     # 對齊 course_table_row 的命名慣例
            "doc_id": doc_id,
            "published_at": None,                # 頁面無明確發布時間
            "fetched_at": now_iso_tpe(),
            "lang": LANG,

            # 索引主要內容（向量/FTS 會吃這欄）
            "text": row_text,

            # 結構化欄位（方便 faceted / 顯示）
            "person_name": person,
            "phone": phone,
            "email": email,
            "office": office,
            "metadata": metadata,
            "checksum_clean": checksum,  # NEW: 用上面算好的,
        }

        append_jsonl(record, args.out)

    print(f"已寫入 JSONL：{args.out}")

if __name__ == "__main__":
    main()
