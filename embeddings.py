# -*- coding: utf-8 -*-
import os, time, requests, logging, math
from dotenv import load_dotenv
import yaml
from pathlib import Path
from paths import ROOT, DATA_DIR, CONFIG_FILE, CACHE_DIR, LOG_DIR, rel, ensure_dir

load_dotenv()
CFG = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))

class GitHubEmbeddings:
    def __init__(self, model: str | None = None) -> None:
        self.model = model or CFG["embed_model"]
        self.token = os.getenv("GITHUB_TOKEN")
        self.url = "https://models.github.ai/inference/embeddings"
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    # --------- 私有：單批請求 ---------
    def _embed_batch(self, texts):
        resp = requests.post(
            self.url,
            headers=self.headers,
            json={"model": self.model, "input": texts},
            timeout=60,
        )
        resp.raise_for_status()
        return [item["embedding"] for item in resp.json()["data"]]

    # --------- 公開 API ---------
    def embed_documents(self, texts, batch_size=64):
        vectors = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            vectors.extend(self._embed_batch(chunk))
            time.sleep(0.3)  # 輕度節流
        return vectors

    def embed_query(self, text):
        return self._embed_batch([text])[0]
