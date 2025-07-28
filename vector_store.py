# -*- coding: utf-8 -*-
import os, shutil, yaml
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document  # 取代 TextLoader
from embeddings import GitHubEmbeddings

CFG = yaml.safe_load(open("config.yaml", encoding="utf-8"))

def load_and_chunk(text: str, source: str) -> list[Document]:
    """將教材文字分段為 chunk 清單"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CFG["chunk_size"],
        chunk_overlap=CFG["chunk_overlap"]
    )
    # 切分並同時注入 metadata；因為只有一段 text，metadatas 也只要一筆，
    # 創出的每個 chunk metadata 都是一樣的
    metadatas = [{"source": source} for _ in range(len(splitter.split_text(text)))]
    docs = splitter.create_documents([text], metadatas=metadatas)
    return docs

def build_or_load(chunks):
    emb = GitHubEmbeddings()
    db_dir = CFG["vector_db_dir"]

    if os.path.exists(db_dir):
        # init 時用 embedding_function 參數
        vectordb = Chroma(
            persist_directory=db_dir,
            embedding_function=emb
        )
        # 新增 chunks
        vectordb.add_documents(chunks)
        return vectordb

    vectordb = Chroma.from_documents(
        documents=chunks, embedding=emb, persist_directory=db_dir
    )
    # vectordb.persist()  # 可選，若你希望每次使用新資料可註解掉
    return vectordb

def reset_db():
    db_dir = CFG["vector_db_dir"]
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)
