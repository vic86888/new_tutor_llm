# chat.py
# -*- coding: utf-8 -*-
import yaml
from tutor_agent import TutorAgent
from vector_store import reset_db  # 若後續需要重置

# 載入設定
CFG = yaml.safe_load(open("config.yaml", encoding="utf-8"))

# 初始化對話代理
agent = TutorAgent()

# 載入已存在的向量庫（假設 persist_directory 在 config.yaml 中）
from langchain_chroma import Chroma
from embeddings import GitHubEmbeddings

emb = GitHubEmbeddings()
vector_dir = CFG["vector_db_dir"]

vectordb = Chroma(
    persist_directory=vector_dir,
    embedding_function=emb
)
print("成功載入向量庫，開始對話")

# 開場摘要
context = "\n\n---\n\n".join(
    [doc.page_content for doc in vectordb.similarity_search("intro", k=CFG["top_k_intro"]) ]
)
from prompt import tutor_guideline, weakness_template
intro_prompt = f"""
以下是今天的教材內容，請先閱讀並列出 2～5 個重點，用親切語氣開場，並詢問學生是否準備好開始學習：
教材內容：
{context}
"""
print("\n助理：", agent.ask(intro_prompt))

# 問答迴圈
from main import multiline_input
while True:
    user_input = multiline_input()
    if user_input.lower() in {"exit", "quit", "bye"}:
        agent.messages.append({"role": "system", "content": "produce_diagnosis"})
        diagnosis = agent.ask(weakness_template)
        print("\n助理（診斷）：\n", diagnosis)
        print("👋 再見！")
        break
    print("\n助理：", agent.ask(user_input))
    
    # 1. 將問題向量化，並找出和問題最匹配的文本一起丟給語言模型，這是我「提問 語言模型回答」在使用的方式
    
    # 2. 我們目前做的家教是「和語言模型對話」，和目前要做的rag技術衝突

    # 3. 目前程式在開場時會以 "intro" 為查詢一次性地從向量庫撈取 2–5 個重點片段，之後所有的使用者提問
    # （agent.ask(user_msg)）並沒有再經過向量化檢索，而是直接把最新的對話歷史（包含系統提示與之前的問答）
    # 傳給模型，無額外拉取新的上下文。

    # 4. 上方為目前的對話
    # 下方為用戶每次的回答都會在經過向量化檢索後，將相關片段與問題一起傳給模型，"無法持續進行對話"

    # 所以prompt必須重新設計！

    '''
        # 1. 根據使用者輸入做相似度搜尋
        relevant = vectordb.similarity_search(user_input, k=CFG["top_k_query"])  # :contentReference[oaicite:5]{index=5}
        # 2. 拼接搜尋到的片段作為上下文
        context = "\n\n---\n\n".join(doc.page_content for doc in relevant)
        # 3. 建立新的 prompt，把上下文與提問一同傳給模型
        qa_prompt = f"""
        以下是與你問題最相關的教材片段，請參考後再回答：

        {context}

        問題：{user_input}
        """
        # 4. 呼叫模型
        print("\n助理：", agent.ask(qa_prompt))
    '''