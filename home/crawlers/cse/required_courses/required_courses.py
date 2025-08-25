# required_courses.py  — 兼容 get_pdf 回傳 dict 或 list；每堂課一行 JSONL
import requests, os, json, argparse, hashlib, datetime, sys
from bs4 import BeautifulSoup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from crawlers.cse.required_courses.get_pdf import extract_text_from_pdf  # 不改你原本的解析規則

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

# NEW: 載入既有 JSONL 的 checksum_clean（若舊資料無此欄位則以 text 重新計算）
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
                    if not cs:  # 兼容舊行
                        cs = sha1_of(obj.get("text", ""))
                    if cs:
                        checksums.add(cs)
                except Exception:
                    continue
    except Exception:
        pass
    return checksums
# END NEW

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=os.path.join("data", "required_courses.jsonl"),
        help="輸出 JSONL 檔案路徑（每行一堂課）"
    )
    args = parser.parse_args()

    # NEW: 讀取既有 checksum，避免重複寫入
    existing_checksums = load_existing_checksums(args.out)

    listing_url = "https://cse.ttu.edu.tw/p/412-1058-2032.php"
    res = requests.get(listing_url)
    soup = BeautifulSoup(res.text, "html.parser")

    pdf_url, title = None, None
    for link in soup.find_all("a", href=True):
        if "大學部學生必修科目表" in link.text:
            pdf_url = "https://cse.ttu.edu.tw" + link["href"]
            title = link.get("title") or link.text.strip()
            print("PDF 路徑：", pdf_url)
            print("標題：", title)
            break
    if not pdf_url:
        print("未找到『大學部學生必修科目表』對應的 PDF 連結。")
        return

    # 兼容：get_pdf 可能回傳 dict 或 list
    parsed = extract_text_from_pdf(pdf_url)

    if isinstance(parsed, dict):
        courses = parsed.get("courses", [])
        notes = parsed.get("notes", [])
    elif isinstance(parsed, list):
        courses = parsed
        notes = []
    else:
        print(f"未知回傳型別：{type(parsed)}，中止。")
        return

    # 將每堂課寫成一行 JSONL；欄位對齊切塊／入庫所需
    site_slug = "required_courses"
    doc_id = sha1_of(site_slug + (pdf_url or ""))  # 穩定文件 ID（同一 PDF 共享）
    fetched_at = now_iso_tpe()

    for r in courses:
        # 來源欄位名沿用 get_pdf 的鍵值；若缺則給預設
        semester = r.get("學期", "")
        category = r.get("類別大項", "")
        sub_category = r.get("子類別", "")
        course_name = r.get("課程名稱", "")
        credit = r.get("學分", 0)

        # 每堂課作為「一列一塊」，row_text 方便之後直接向量化/FTS5
        row_text = "｜".join([
            semester, category, sub_category, course_name, f"{credit} 學分"
        ]).strip("｜")

        # NEW: 寫入前先檢查 checksum；存在就跳過
        checksum = sha1_of(row_text)
        if checksum in existing_checksums:
            continue
        existing_checksums.add(checksum)
        # END NEW

        record = {
            # —— 統一的 cleaned JSONL 欄位（之後可直接切塊或視為已切塊）——
            "site_slug": site_slug,
            "url": listing_url,         # 列表頁
            "pdf_url": r.get("來源PDF", pdf_url),
            "title": title,
            "source_kind": "pdf",
            "type": "course_table_row",  # 每行就是一個「表格列」
            "doc_id": doc_id,            # 同一 PDF 共用 doc_id（方便 doc 級去重）
            "published_at": None,        # 有年份再補
            "fetched_at": fetched_at,
            "lang": "zh-Hant",

            # 內容：一列即一段文字
            "text": row_text,

            # 保留結構化欄位（方便 faceted 檢索/排序）
            "semester": semester,
            "category": category,
            "sub_category": sub_category,
            "course_name": course_name,
            "credit": credit,

            # 供去重／變更偵測
            "checksum_clean": checksum,  # NEW: 用上面算好的
        }

        append_jsonl(record, args.out)

    # 若需要，也可把備註寫成類型為 note 的行（可選）
    for note in notes:
        note_text = str(note).strip()
        if not note_text:
            continue

    # NEW: 備註也做 checksum 檢查
        note_checksum = sha1_of(note_text)
        if note_checksum in existing_checksums:
            continue
        existing_checksums.add(note_checksum)
        # END NEW

        append_jsonl({
            "site_slug": site_slug,
            "url": listing_url,
            "pdf_url": pdf_url,
            "title": f"{title}（備註）",
            "source_kind": "pdf",
            "type": "course_table_note",
            "doc_id": doc_id,
            "published_at": None,
            "fetched_at": fetched_at,
            "lang": "zh-Hant",
            "text": note_text,
            "checksum_clean": note_checksum,  # NEW
        }, args.out)

    print(f"已寫入 JSONL：{args.out}")

if __name__ == "__main__":
    main()
