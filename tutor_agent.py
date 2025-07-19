# -*- coding: utf-8 -*-
import os, time, yaml, logging, requests
from dotenv import load_dotenv
from prompt import tutor_guideline

load_dotenv()
CFG = yaml.safe_load(open("config.yaml", encoding="utf-8"))

class TutorAgent:
    def __init__(self):
        self.messages = [{"role": "system", "content": tutor_guideline}]
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
    def ask(self, user_msg: str) -> str:
        self._append_history("user", user_msg)
        self._trim_history()

        for attempt in range(self.retry):
            try:
                resp = requests.post(
                    self.url,
                    headers=self.headers,
                    json={"model": self.model_name, "messages": self.messages},
                    timeout=60,
                )
                resp.raise_for_status()
                break
            except requests.exceptions.HTTPError as e:
                logging.warning(f"HTTPError {e} (try {attempt+1}/{self.retry})")
                if attempt == self.retry - 1:
                    raise
                time.sleep(2)

        reply = resp.json()["choices"][0]["message"]["content"]
        self._append_history("assistant", reply)
        return reply

    # --------- 私有 ---------
    def _append_history(self, role, content):
        self.messages.append({"role": role, "content": content.strip()})

    def _trim_history(self):
        max_len = self.max_history * 2 + 1
        if len(self.messages) > max_len:
            self.messages = self.messages[:1] + self.messages[-max_len + 1 :]
