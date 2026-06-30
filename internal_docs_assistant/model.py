from __future__ import annotations

import hashlib
import json
import math
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class ModelConfig:
    """Names of the Gemini models used by the app."""

    generation_model: str = os.getenv("GEMINI_GENERATION_MODEL", "gemini-3.5-flash")
    embedding_model: str = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "openrouter/free")
    openrouter_base_url: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


class GeminiModel:
    """Small Gemini API wrapper.

    If no Gemini key is present, the methods return simple local fallbacks so the
    rest of the project can still be tested.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("Gemini_API_Key")
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.client = None
        self.types = None

        if self.api_key:
            from google import genai
            from google.genai import types

            self.client = genai.Client(api_key=self.api_key)
            self.types = types

    @property
    def online(self) -> bool:
        return self.client is not None or self.openrouter_api_key is not None

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            if self.client:
                try:
                    embeddings.append(self._gemini_embedding(text))
                    continue
                except Exception:
                    pass
            embeddings.append(self._local_embedding(text))
        return embeddings

    def json_chat(self, system_prompt: str, user_prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        if self.client and self.types:
            try:
                response = self.client.models.generate_content(
                    model=self.config.generation_model,
                    contents=user_prompt,
                    config=self.types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        response_mime_type="application/json",
                        response_schema=schema,
                    ),
                )
                return json.loads(response.text or "{}")
            except Exception:
                pass

        if self.openrouter_api_key:
            content = self._openrouter_chat(system_prompt, user_prompt, json_mode=True)
            try:
                return json.loads(content or "{}")
            except json.JSONDecodeError:
                return {}

        return {}

    def text_chat(self, system_prompt: str, user_prompt: str) -> str:
        if self.client and self.types:
            try:
                response = self.client.models.generate_content(
                    model=self.config.generation_model,
                    contents=user_prompt,
                    config=self.types.GenerateContentConfig(system_instruction=system_prompt),
                )
                return response.text or ""
            except Exception:
                pass

        if self.openrouter_api_key:
            return self._openrouter_chat(system_prompt, user_prompt)

        return ""

    def _openrouter_chat(self, system_prompt: str, user_prompt: str, json_mode: bool = False) -> str:
        payload: dict[str, Any] = {
            "model": self.config.openrouter_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        request = urllib.request.Request(
            f"{self.config.openrouter_base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost"),
                "X-OpenRouter-Title": os.getenv("OPENROUTER_APP_TITLE", "Internal Docs Assistant"),
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            return ""

        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return str(message.get("content") or "")

    def _gemini_embedding(self, text: str) -> list[float]:
        response = self.client.models.embed_content(
            model=self.config.embedding_model,
            contents=text,
        )
        if not response.embeddings:
            return []
        return list(response.embeddings[0].values)

    @staticmethod
    def _local_embedding(text: str) -> list[float]:
        vector = [0.0] * 256
        words = text.lower().split()

        for word in words:
            digest = hashlib.sha256(word.encode("utf-8")).hexdigest()
            bucket = int(digest, 16) % len(vector)
            vector[bucket] += 1.0

        length = math.sqrt(sum(value * value for value in vector))
        if length == 0:
            return vector

        return [value / length for value in vector]


ApiModel = GeminiModel
