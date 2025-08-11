# extract_from_mysql.py
# -*- coding: utf-8 -*-
import mysql.connector
import pandas as pd
import numpy as np
import os
import datetime
import json
import sys

# === 連線到 phpMyAdmin 對應的 MySQL 資料庫 ===
try:
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="test"  # 改成你的資料庫
    )
except mysql.connector.Error as err:
    print(f"❌ 資料庫連線失敗：{err}")
    sys.exit(1)

# === 讀取商品資料表 ===
try:
    query = "SELECT * FROM laptops;"  # 改成你的資料表
    df = pd.read_sql(query, conn)
finally:
    conn.close()

if df.empty:
    print("⚠️ 查詢結果為空，請確認資料表有資料")
    sys.exit(0)

# === 清理資料 ===
# 1) 把 ±inf 變成 NaN
df = df.replace([np.inf, -np.inf], np.nan)

# 2) 轉成 object 型別，將 NaN 改成 None（輸出時為 JSON null）
df = df.astype(object).where(pd.notnull(df), None)

# 3) （可選）檢查是否仍有 NaN
if df.applymap(lambda x: isinstance(x, float) and (x != x)).any().any():
    print("❌ 清理後仍有 NaN，請檢查資料")
    sys.exit(1)

# === 準備輸出資料夾與檔名 ===
os.makedirs("data", exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
json_output_path = f"data/products_{timestamp}.json"

# === 轉成 JSON 並輸出 ===
records = df.to_dict(orient="records")
with open(json_output_path, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2, allow_nan=False)

print(f"✅ 已將 {len(df)} 筆商品資料寫入 JSON：{json_output_path}")
print("👉 接下來可轉為向量或提供給前端使用")
