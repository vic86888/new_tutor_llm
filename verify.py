# verify.py
# -*- coding: utf-8 -*-
import os
import yaml
import json
import time
import logging
import requests
from dotenv import load_dotenv
from prompt import tutor_guideline

load_dotenv()
CFG = yaml.safe_load(open("config.yaml", encoding="utf-8"))

class VerifierAgent:
    def __init__(self):
        # 使用 GitHub Model API
        self.system_msg = tutor_guideline
        self.retry = CFG.get("retry", 3)
        self.model_name = CFG.get("verify_model_name", CFG.get("model_name"))
        self.headers = {
            "Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        self.url = "https://models.github.ai/inference/chat/completions"

    def verify(self, query: str, answer: str, context: str) -> dict:
        """
        輸入：query, answer, context
        輸出：含 scoring 與 pass/fail 的 dict
        """
        verify_prompt = f"""
你是專家級的「驗證者模型」。請依以下標準評估回答是否合格，並以 JSON 格式回傳：

【輸入】
問題: {query}
回答: {answer}
參考資料:
{context}

評分標準：
1. 正確性 (1-5)
2. 完整性 (1-5)
3. 相關性 (1-5)
4. 清晰度 (1-5)

請輸出 JSON:
{{
  "scores": {{
    "correctness": x,
    "completeness": y,
    "relevance": z,
    "clarity": w
  }},
  "total_score": t,
  "pass": true|false,
  "comments": "綜合評語…"
}}
"""
        # 組建 messages
        messages = [
            {"role": "system", "content": self.system_msg},
            {"role": "user",   "content": verify_prompt}
        ]

        # 重試機制
        for attempt in range(self.retry):
            try:
                resp = requests.post(
                    self.url,
                    headers=self.headers,
                    json={"model": self.model_name, "messages": messages},
                    timeout=60,
                )
                resp.raise_for_status()
                break
            except requests.exceptions.HTTPError as e:
                logging.warning(f"Verifier HTTPError {e} (try {attempt+1}/{self.retry})")
                if attempt == self.retry - 1:
                    raise
                time.sleep(2)

        content = resp.json().get("choices", [])[0].get("message", {}).get("content", "")
        # 嘗試解析 JSON
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {"raw": content}

# 單一函式接口，方便在 chat.py 中調用
verifier = VerifierAgent()

def verify_answer(query: str, answer: str, context: str) -> dict:
    return verifier.verify(query, answer, context)
