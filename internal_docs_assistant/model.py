from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import Any


@dataclass
class ModelConfig:
    """Names of the Gemini models used by the app."""

    generation_model: str = os.getenv("GEMINI_GENERATION_MODEL", "gemini-3.5-flash")
    embedding_model: str = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")


class GeminiModel:
    """Small Gemini API wrapper.

    If no Gemini key is present, the methods return simple local fallbacks so the
    rest of the project can still be tested.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("Gemini_API_Key")
        self.client = None
        self.types = None

        if self.api_key:
            from google import genai
            from google.genai import types

            self.client = genai.Client(api_key=self.api_key)
            self.types = types

    @property
    def online(self) -> bool:
        return self.client is not None

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            if self.client:
                embeddings.append(self._gemini_embedding(text))
            else:
                embeddings.append(self._local_embedding(text))
        return embeddings

    def json_chat(self, system_prompt: str, user_prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        if not self.client or not self.types:
            return {}

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

    def text_chat(self, system_prompt: str, user_prompt: str) -> str:
        if not self.client or not self.types:
            return ""

        response = self.client.models.generate_content(
            model=self.config.generation_model,
            contents=user_prompt,
            config=self.types.GenerateContentConfig(system_instruction=system_prompt),
        )
        return response.text or ""

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
