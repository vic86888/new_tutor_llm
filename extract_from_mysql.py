# extract_from_mysql.py
# -*- coding: utf-8 -*-
import mysql.connector
import pandas as pd
import os
import datetime
import json

# === 連線到 phpMyAdmin 對應的 MySQL 資料庫 ===
try:
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="nvidia_gpu_comparison"
    )
except mysql.connector.Error as err:
    print(f"❌ 資料庫連線失敗：{err}")
    exit()

# === 讀取商品資料表 ===
try:
    query = "SELECT * FROM nvidia_gpu_comparison;"  # 資料表名稱若為 products 請改成 products
    df = pd.read_sql(query, conn)
    conn.close()
except Exception as e:
    print(f"❌ 查詢資料時出錯：{e}")
    exit()

if df.empty:
    print("⚠️ 查詢結果為空，請確認資料表有資料")
    exit()

# === 將 DataFrame 轉為 JSON 並儲存 ===
os.makedirs("data", exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
json_output_path = f"data/products_{timestamp}.json"

with open(json_output_path, "w", encoding="utf-8") as f:
    json.dump(df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

print(f"✅ 已將 {len(df)} 筆商品資料寫入 JSON：{json_output_path}")
print("👉 接下來可轉為向量或提供給前端使用")
