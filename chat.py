# chat.py
# -*- coding: utf-8 -*-
import yaml
from sentence_transformers import CrossEncoder # 用於重排模型
from langchain_community.chat_message_histories import ChatMessageHistory
from tutor_agent import TutorAgent
from vector_store import reset_db  # 若後續需要重置
import verify  # 新增：驗證模組

# 選一個合適的重排模型
# ms-marco-MiniLM-L-6-v2 在速度與效果間有不錯的平衡
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

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
# 在這裡新增
history = ChatMessageHistory()
agent = TutorAgent()

# 開場摘要
# context = "\n\n---\n\n".join(
#     [doc.page_content for doc in vectordb.similarity_search("intro", k=CFG["top_k_intro"]) ]
# )
from prompt import tutor_guideline
# intro_prompt = f"""
# 系統提示：{tutor_guideline}

# 請使用者提出和參考資料相關的問題：
# 參考資料：
# {context}
# """
# print("\n助理：", agent.ask(intro_prompt))
print("\n助理：", "歡迎使用知識問答系統！請輸入您的問題，我會根據提供的產品資訊回答。")


# 問答迴圈
from main import multiline_input
while True:
    user_input = multiline_input()
    if user_input.lower() in {"exit", "quit", "bye"}:
        print("👋 再見！")
        break
    # 1. 根據使用者問題做相似度搜尋
    relevant_docs = vectordb.similarity_search(
        user_input,
        k=CFG.get("top_k_query", 5)
    )
    context = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])

     # 1.5 加入重排 (re‑ranking)
    #  1) 準備 (query, doc_text) pair list
    pairs = [
        (user_input, doc.page_content)
        for doc in relevant_docs
    ]
    #  2) 用 CrossEncoder 預測每一對的相關度
    scores = cross_encoder.predict(pairs)
    # 3. 把 doc 跟 score 打包成 list of tuples
    doc_score_pairs = list(zip(relevant_docs, scores))

    # 4. 依 score 排序（由大到小）
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

    # 取重排後分數最高的前 N 個 chunk
    top_k = CFG.get("top_k_rerank", 3)
    selected = doc_score_pairs[:top_k]

    # —— 在這裡印出本次用到的 chunk ——  
    print("\n本次使用的 chunk (已依相關度排序)：")
    for idx, (doc, score) in enumerate(selected, start=1):
        # 假設你在建立向量庫時有把 source 記在 metadata 裡
        source = doc.metadata.get("source", "unknown")
        snippet = doc.page_content.replace("\n", " ")[:100]  # 取前100字
        print(f"{idx}. [score={score:.3f}] 來源: {source}，內容預覽: {snippet}…")

    # 5. 產生排序後的 context（可同時顯示分數）
    context = "\n\n---\n\n".join(
        f"[score={score:.3f}] {doc.page_content}"
        for doc, score in selected
    )

    # 6. 把排序後的 documents 拆回來，如果後續需要再操作
    sorted_docs = [doc for doc, _ in doc_score_pairs]

    # 2. 取用對話記憶
    history_msgs = history.messages
    history_text = "\n".join([f"{msg.type}: {msg.content}" for msg in history_msgs])

    prompt = f"""
        ### ➤ System
        {tutor_guideline}

        ### ➤ Memory
        {history_text}

        ### ➤ 下面是排序後的參考資料（每段包含 reranker 給的相關性分數與來源），請只根據這些資料回答問題，不要憑空補充未出現在資料中的內容。差異高的分數代表內容比較匹配，但最終答案應以文字內容為依據。
        {context}

        ### ➤ User
        {user_input}
        """


    # 4. 呼叫模型並印出回答
    answer = agent.ask(prompt)
    print("\n助理：", answer)

    # 5. 更新對話記憶
    history.add_user_message(user_input)
    history.add_ai_message(answer)

    # 6. 驗證階段
    report = verify.verify_answer(user_input, answer, context)
    print("\n🔍 驗證報告：", report)
    
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