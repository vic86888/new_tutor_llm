import os, json, requests
import shutil
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.schema import Document
from langchain_chroma import Chroma
import prompt

# 📚 Step 1: 切片
def load_and_chunk(filepath, chunk_size=1000, overlap=100):
    loader = TextLoader(filepath, encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_documents(docs)

# ⚙️ Step 2: 產生 embeddings
class GitHubEmbeddings:
    def __init__(self, model="openai/text-embedding-3-small"):
        load_dotenv()
        self.model = model
        self.token = os.getenv("GITHUB_TOKEN")
        self.url = "https://models.github.ai/inference/embeddings"
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
        }

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        resp = requests.post(self.url, headers=self.headers, json={"model": self.model, "input": texts})
        resp.raise_for_status()
        return [item["embedding"] for item in resp.json()["data"]]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

# 🗄 Step 3: 建立向量庫
def build_vector_store(chunks):
    emb = GitHubEmbeddings()
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=emb,
        persist_directory="chroma_db"
    )
    # vectordb.persist()
    return vectordb

# 💬 Step 4: 使用 Chat 模型整合檢索結果
def chat_with_context(chunks, user_input, messages):
    messages.append({"role": "user", "content": user_input})

    # 呼叫 Chat 接口
    load_dotenv()
    token = os.getenv("GITHUB_TOKEN")
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/json"
    }
    url = "https://models.github.ai/inference/chat/completions"

    resp = requests.post(url, headers=headers, json={
        "model": "openai/gpt-4.1-mini",
        "messages": messages
    })
    resp.raise_for_status()

    reply = resp.json()["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": reply})
    return reply

# ▶️ 主流程
if __name__ == "__main__":
    # 清除舊的向量庫
    if os.path.exists("chroma_db"):
        shutil.rmtree("chroma_db")

    messages = [{"role": "system", "content": prompt.tutor_guideline}]
    chunks = load_and_chunk(r"C:\gpt_tutor\py4e_3.txt")
    print(f"已切出 {len(chunks)} 片")

    vectordb = build_vector_store(chunks)
    print("✔️ 向量索引建立完成")

    print("輸入問題，輸入 exit 離開。")

    # 把教材摘要出來（用 similarity_search 抓幾段內容）
    results = vectordb.similarity_search("Python", k=3)
    context = "\n\n---\n\n".join([doc.page_content for doc in results])

    # 把教材內容放進開場訊息中
    intro_question = f"""
    當你收到教材後，請先閱讀教材內容，歸納出 2～5 個學生今天要學習的重點，並用簡單清楚的話在開場時告訴學生：
    「我們今天會學到什麼」。請列點或條列方式呈現，幫助學生建立學習期待。
    然後和學生確認是否準備開始學習

    教材內容：
    {context}
    """

    first_question = chat_with_context(vectordb, intro_question, messages)
    print("\n助理：", first_question)

    while True:
        q = input("\n你：").strip()
        if q.lower() in ("exit","quit"):
            print("👋 再見！"); break
        answer = chat_with_context(vectordb, q, messages)
        print("\n助理：", answer)
