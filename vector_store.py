# -*- coding: utf-8 -*-
import os, shutil, yaml
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document  # 取代 TextLoader
from embeddings import GitHubEmbeddings

CFG = yaml.safe_load(open("config.yaml", encoding="utf-8"))

def load_and_chunk(text: str):
    """將教材文字分段為 chunk 清單"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CFG["chunk_size"],
        chunk_overlap=CFG["chunk_overlap"]
    )
    docs = splitter.create_documents([text])
    return docs

def build_or_load(chunks):
    emb = GitHubEmbeddings()
    db_dir = CFG["vector_db_dir"]

    if os.path.exists(db_dir):
        return Chroma(persist_directory=db_dir, embedding=emb)

    vectordb = Chroma.from_documents(
        documents=chunks, embedding=emb, persist_directory=db_dir
    )
    # vectordb.persist()  # 可選，若你希望每次使用新資料可註解掉
    return vectordb

def reset_db():
    db_dir = CFG["vector_db_dir"]
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)
