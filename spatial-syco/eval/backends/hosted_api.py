"""
Hosted-API backend: run VLMs through any OpenAI-compatible endpoint with NO local GPU.

Works with Together / OpenRouter / DeepInfra / OpenAI / Google (OpenAI-compat) by setting
BASE_URL + API key. This is the fastest path to your first Week-4 baseline numbers:
a 40-item pilot costs cents and needs no HPC account.

Env vars:
    SYCO_API_BASE   e.g. https://api.together.xyz/v1
                         https://openrouter.ai/api/v1
                         https://api.deepinfra.com/v1/openai
                         https://api.openai.com/v1
    SYCO_API_KEY    your key  (NEVER commit this — it's in .gitignore via .env)

Drop this in eval/backends/hosted_api.py and import it from run_eval.py's make_backend().
"""
from __future__ import annotations
import base64
import mimetypes
import os
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; fall back to shell env vars


class HostedAPIBackend:
    def __init__(self, model: str):
        from openai import OpenAI  # local import so API-less runs don't need the package
        self.name = model
        self.model = model
        base = os.environ.get("SYCO_API_BASE", "https://api.openai.com/v1")
        key = os.environ.get("SYCO_API_KEY")
        if not key:
            raise RuntimeError("set SYCO_API_KEY (and SYCO_API_BASE) in your environment / .env")
        self.client = OpenAI(base_url=base, api_key=key)

    @staticmethod
    def _encode_image(path: str) -> str:
        mime = mimetypes.guess_type(path)[0] or "image/jpeg"
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f"data:{mime};base64,{b64}"

    def generate(self, system: str, user: str, image_path: str, item: dict,
                history: list | None = None) -> str:
        """`history`: optional prior turns (list of {"role","content"} dicts, text-only)
        inserted between the system prompt and the new user turn -- used for the
        misleading_followup sustained-pressure sub-experiment (see eval/run_eval.py)."""
        content = [{"type": "text", "text": user}]
        if image_path and os.path.isfile(image_path):
            content.append({"type": "image_url",
                            "image_url": {"url": self._encode_image(image_path)}})

        messages = [{"role": "system", "content": system}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": content})

        from openai import APIConnectionError, APITimeoutError

        max_attempts = 8
        for attempt in range(max_attempts):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.0,   # deterministic; raise + repeat for Consistency@k
                    max_tokens=256,
                    timeout=60,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                status = getattr(getattr(e, "response", None), "status_code", None)
                retryable = status in (429, 500, 502, 503) or isinstance(
                    e, (APIConnectionError, APITimeoutError))
                if attempt < max_attempts - 1 and retryable:
                    delay = min(2 ** attempt, 60)   # 1,2,4,8,16,32,60,60s
                    time.sleep(delay)
                    continue
                raise


if __name__ == "__main__":
    # smoke test (requires env vars + a real image path)
    b = HostedAPIBackend(os.environ.get("SYCO_MODEL", "gpt-4o-mini"))
    print("backend ready:", b.name)
