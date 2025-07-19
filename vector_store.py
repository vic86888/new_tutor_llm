# -*- coding: utf-8 -*-
import os, shutil, yaml
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from embeddings import GitHubEmbeddings

CFG = yaml.safe_load(open("config.yaml", encoding="utf-8"))

def load_and_chunk(path: str):
    loader = TextLoader(path, encoding="utf-8")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CFG["chunk_size"], chunk_overlap=CFG["chunk_overlap"]
    )
    return splitter.split_documents(docs)

def build_or_load(chunks):
    emb = GitHubEmbeddings()
    db_dir = CFG["vector_db_dir"]

    if os.path.exists(db_dir):
        return Chroma(persist_directory=db_dir, embedding=emb)

    vectordb = Chroma.from_documents(
        documents=chunks, embedding=emb, persist_directory=db_dir
    )
    # vectordb.persist()
    return vectordb

def reset_db():
    db_dir = CFG["vector_db_dir"]
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)
