# 智能商品機器人專案

本專案以 **Python + 向量資料庫** 為核心，透過 RAG（Retrieval-Augmented Generation）技術，打造可對話的商品推薦／諮詢機器人。下列說明涵蓋各程式檔案的用途與關係，協助快速瞭解專案架構與開發流程。

---

## 專案目錄結構(主要功能的檔案)
```text
.
├─ chat.py
├─ ingest.py
├─ main.py
├─ vector_store.py
├─ embedding.py
├─ tutor_agent.py
├─ verify.py
├─ inspect_vectordb.py
├─ config.yaml
└─ data/
   └─ …         # 存放欲轉入向量庫的原始檔案


chat.py 
執行後即可開始和智能商品機器人對話
ingest.py 
執行後，如果同目錄的data資料夾中有*剛好一個符合格式的檔案*，會將其切割後轉換向量存入向量資料庫
main.py 
目前只有引用其中的函式使用
vector_store.py
在這邊會將檔案切割成chunk，並將每個chunk轉換為向量儲存到向量資料庫
embedding.py
將chunk轉換為向量的程式碼
tutor_agent.py 
在這裡將重組過後的新prompt提供給語言模型，獲得語言模型的回答
verify.py
使用另一個模型對語言模型的回答根據各種標準進行評分(這方法還沒有說服力，後續可能不採用)
config.yaml
裡面放著各種檔案使用的重要參數
inspect_vectordb.py
可以查看目前所有的切塊內容和向量
