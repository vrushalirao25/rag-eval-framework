"""Thin provider abstraction so the generator and the judge can use different
model families (recommended, to avoid a judge favouring its own outputs).

Providers:
  mock     – no LLM; the assistant and judge fall back to deterministic
             heuristics so the pipeline runs offline and in CI.
  bedrock  – Amazon Bedrock Converse API (boto3).
  openai   – OpenAI Chat Completions API.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod

log = logging.getLogger("evalkit.llm")

# Only these are worth retrying; auth, validation and access errors fail immediately.
TRANSIENT_AWS_CODES = {"ThrottlingException", "ServiceUnavailableException", "ModelNotReadyException",
                       "InternalServerException", "ModelTimeoutException"}
TRANSIENT_EXC_NAMES = {"RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError",
                       "ReadTimeoutError", "ConnectTimeoutError", "EndpointConnectionError"}


def is_transient(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    code = response.get("Error", {}).get("Code") if isinstance(response, dict) else None
    return code in TRANSIENT_AWS_CODES or type(exc).__name__ in TRANSIENT_EXC_NAMES


class LLMClient(ABC):
    name: str

    @abstractmethod
    def complete(self, system: str, prompt: str, max_tokens: int = 800) -> str:
        ...

    def complete_with_retry(self, system: str, prompt: str, max_tokens: int = 800, retries: int = 3) -> str:
        for attempt in range(retries + 1):
            try:
                return self.complete(system, prompt, max_tokens)
            except Exception as exc:  # provider SDKs raise many exception types
                if not is_transient(exc) or attempt == retries:
                    raise
                wait = 2 ** attempt
                log.warning("transient error from %s (%s); retry %d/%d in %ss",
                            self.name, type(exc).__name__, attempt + 1, retries, wait)
                time.sleep(wait)
        raise AssertionError("unreachable")

    def ping(self) -> None:
        """Preflight: one minimal real call. Raises if credentials, region or model access are wrong."""
        if not self.complete("Reply with the single word OK.", "ping", max_tokens=5).strip():
            raise RuntimeError(f"{self.name} returned an empty response to the preflight check")


class BedrockClient(LLMClient):
    def __init__(self, model_id: str, region: str | None = None):
        import boto3  # imported lazily so mock mode needs no AWS SDK

        self.model_id = model_id
        self.name = f"bedrock:{model_id}"
        region = region or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
        if not region:
            raise ValueError("Set AWS_REGION (no default region is assumed)")
        self._client = boto3.client("bedrock-runtime", region_name=region)

    def complete(self, system: str, prompt: str, max_tokens: int = 800) -> str:
        resp = self._client.converse(
            modelId=self.model_id,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"temperature": 0.0, "maxTokens": max_tokens},
        )
        return "".join(block.get("text", "") for block in resp["output"]["message"]["content"])


class OpenAIClient(LLMClient):
    def __init__(self, model: str):
        from openai import OpenAI

        self.model = model
        self.name = f"openai:{model}"
        self._client = OpenAI()

    def complete(self, system: str, prompt: str, max_tokens: int = 800) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content or ""


def get_client(provider: str, model: str | None) -> LLMClient | None:
    provider = (provider or "mock").lower()
    if provider == "mock":
        return None
    if not model or model.startswith("$"):
        raise ValueError(f"Provider '{provider}' needs a model ID (check your .env / config)")
    if provider == "bedrock":
        return BedrockClient(model)
    if provider == "openai":
        return OpenAIClient(model)
    raise ValueError(f"Unknown provider: {provider}")


def extract_json(text: str) -> dict | None:
    """Parse the first JSON object in a model response (tolerates ```json fences)."""
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    while start != -1:
        depth = 0
        for end in range(start, len(cleaned)):
            if cleaned[end] == "{":
                depth += 1
            elif cleaned[end] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(cleaned[start : end + 1])
                        return obj if isinstance(obj, dict) else None
                    except json.JSONDecodeError:
                        break
        start = cleaned.find("{", start + 1)
    return None
