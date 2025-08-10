# -*- coding: utf-8 -*-
import os, time, yaml, logging, requests
from dotenv import load_dotenv
from prompt import tutor_guideline
from pathlib import Path
from paths import ROOT, DATA_DIR, CONFIG_FILE, CACHE_DIR, LOG_DIR, rel, ensure_dir

load_dotenv()
CFG = yaml.safe_load(CONFIG_FILE.read_text(encoding="utf-8"))

class TutorAgent:
    def __init__(self):
        # 取代 self.messages，改成只儲存系統提示
        self.system_msg = tutor_guideline
        self.max_history = CFG["max_history"]
        self.retry = CFG["retry"]
        self.model_name = CFG["model_name"]
        self.headers = {
            "Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        self.url = "https://models.github.ai/inference/chat/completions"

    # --------- 公開方法 ---------
    def ask(self, prompt: str) -> str:
        """
        根據單次 prompt（已包含系統提示、檢索上下文、對話摘要等）呼叫模型，並回傳回答
        """
        # 即時組建 messages：只含一條 system 與一條 user
        messages = [
            {"role": "system", "content": self.system_msg},
            {"role": "user",   "content": prompt.strip()}
        ]

        # 重試機制不變
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
                logging.warning(f"HTTPError {e} (try {attempt+1}/{self.retry})")
                if attempt == self.retry - 1:
                    raise
                time.sleep(2)

        # 回傳模型結果
        return resp.json()["choices"][0]["message"]["content"]
