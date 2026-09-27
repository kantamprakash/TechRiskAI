"""LLM clients.

`OpenRouterLLM` calls models through OpenRouter's OpenAI-compatible API and reads each
model's context length / max output tokens from OpenRouter's model catalog.
`MockLLM` is a keyword-based stand-in so the pipeline can be run and tested without an API key.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from difflib import get_close_matches
from functools import lru_cache
from typing import Protocol

from .config import CHARS_PER_TOKEN, Settings

log = logging.getLogger(__name__)

DEFAULT_CONTEXT_TOKENS = 128_000
DEFAULT_MAX_OUTPUT_TOKENS = 8_192
PROMPT_OVERHEAD_TOKENS = 2_000  # system prompt + project context + safety margin


class LLMError(RuntimeError):
    pass


@dataclass
class ModelInfo:
    id: str
    context_length: int | None = None
    max_completion_tokens: int | None = None
    supports_json_mode: bool | None = None
    prompt_price_per_m: float | None = None      # USD per 1M input tokens
    completion_price_per_m: float | None = None  # USD per 1M output tokens


@dataclass
class Sizing:
    """Resolved context/chunk/output budget for a run."""
    context_tokens: int
    max_output_tokens: int
    chunk_tokens: int
    overlap_tokens: int

    @property
    def chunk_chars(self) -> int:
        return self.chunk_tokens * CHARS_PER_TOKEN

    @property
    def overlap_chars(self) -> int:
        return self.overlap_tokens * CHARS_PER_TOKEN


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    models_used: Counter = field(default_factory=Counter)
    switches: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, model: str, prompt: int, completion: int, cost: float) -> None:
        with self._lock:
            self.calls += 1
            self.prompt_tokens += prompt
            self.completion_tokens += completion
            self.cost_usd += cost
            self.models_used[model] += 1

    def add_switch(self, message: str) -> None:
        with self._lock:
            self.switches.append(message)


class LLM(Protocol):
    name: str
    usage: Usage

    def model_info(self) -> ModelInfo: ...

    def chat_json(self, task: str, system: str, user: str, max_tokens: int) -> dict: ...


def resolve_sizing(cfg: Settings, info: ModelInfo) -> Sizing:
    """Explicit settings win; otherwise derive from the model's OpenRouter metadata."""
    context = cfg.llm_context_tokens or info.context_length or DEFAULT_CONTEXT_TOKENS
    model_out_cap = info.max_completion_tokens or DEFAULT_MAX_OUTPUT_TOKENS
    max_out = cfg.llm_max_output_tokens or min(DEFAULT_MAX_OUTPUT_TOKENS, model_out_cap)
    max_out = min(max_out, model_out_cap, context // 2)

    room = context - max_out - PROMPT_OVERHEAD_TOKENS - cfg.rag_prompt_tokens
    if room < 1_000:
        raise LLMError(f"Context of {context} tokens is too small for {max_out} output tokens; "
                       f"lower LLM_MAX_OUTPUT_TOKENS or choose a model with a larger context")
    chunk = cfg.chunk_tokens or min(cfg.max_chunk_tokens, int(room * 0.8))
    if chunk > room:
        log.warning("CHUNK_TOKENS=%s does not fit the model context (%s); using %s", chunk, context, room)
        chunk = room
    overlap = min(cfg.chunk_overlap_tokens, chunk // 4)
    return Sizing(context, max_out, chunk, overlap)


def extract_json(text: str) -> dict:
    """Parse a JSON object out of a model reply that may include fences or chatter."""
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.MULTILINE).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            ls, le = text.find("["), text.rfind("]")
            if ls != -1 and le > ls:
                return {"risks": json.loads(text[ls:le + 1])}
            raise LLMError(f"No JSON object in model reply: {text[:200]!r}")
        candidate = re.sub(r",\s*([}\]])", r"\1", text[start:end + 1])  # trailing commas
        value = json.loads(candidate)
    if isinstance(value, list):
        return {"risks": value}
    if not isinstance(value, dict):
        raise LLMError("Model reply JSON is not an object")
    return value


def _get_json(url: str, headers: dict | None = None) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.load(resp)


@lru_cache(maxsize=4)
def fetch_model_catalog(base_url: str) -> dict[str, dict]:
    """OpenRouter's public model list, keyed by model id."""
    return {m["id"]: m for m in _get_json(f"{base_url.rstrip('/')}/models").get("data", [])}


def _price_per_m(value) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v * 1_000_000 if v >= 0 else None


def is_free_model(model: str, catalog: dict[str, dict] | None = None) -> bool:
    if model.endswith(":free") or model == "openrouter/free":
        return True
    pricing = ((catalog or {}).get(model) or {}).get("pricing") or {}
    return pricing.get("prompt") == "0" and pricing.get("completion") == "0"


class ModelBlocks:
    """Process-wide record of models (or whole groups: all free / all paid) that are
    temporarily unusable, so later calls and later runs skip them until the block expires."""

    FREE, PAID = "__all_free__", "__all_paid__"

    def __init__(self):
        self._until: dict[str, float] = {}
        self._reason: dict[str, str] = {}
        self._lock = threading.Lock()

    def block(self, key: str, seconds: float, reason: str) -> None:
        with self._lock:
            self._until[key] = max(self._until.get(key, 0), time.time() + seconds)
            self._reason[key] = reason

    def blocked(self, model: str, free: bool) -> str | None:
        """Reason the model is blocked, or None if usable."""
        now = time.time()
        with self._lock:
            for key in (model, self.FREE if free else self.PAID):
                if self._until.get(key, 0) > now:
                    return self._reason[key]
        return None

    def seconds_until_available(self, model: str, free: bool) -> float:
        now = time.time()
        with self._lock:
            return max(0.0, max(self._until.get(k, 0) for k in (model, self.FREE if free else self.PAID)) - now)

    def clear(self) -> None:
        with self._lock:
            self._until.clear()
            self._reason.clear()


MODEL_BLOCKS = ModelBlocks()


def _error_details(e: Exception) -> tuple[int | None, str, float | None]:
    """(HTTP status, lower-cased message, seconds until the rate limit resets) from an OpenAI SDK error."""
    status = getattr(e, "status_code", None)
    body = getattr(e, "body", None)
    message = str(e)
    reset_ms = None
    if isinstance(body, dict):
        err = body.get("error", body) if isinstance(body.get("error", body), dict) else body
        message = f"{message} {err.get('message', '')} {json.dumps(err.get('metadata') or {})}"
        reset_ms = ((err.get("metadata") or {}).get("headers") or {}).get("X-RateLimit-Reset")
    response = getattr(e, "response", None)
    if reset_ms is None and response is not None:
        reset_ms = response.headers.get("x-ratelimit-reset")
    wait = None
    if reset_ms:
        try:
            wait = max(0.0, int(reset_ms) / 1000 - time.time())
        except ValueError:
            pass
    return status, message.lower(), wait


def _seconds_to_utc_midnight() -> float:
    now = time.time()
    return 86_400 - (now % 86_400)


class OpenRouterLLM:
    """OpenRouter client with automatic model switching.

    The chain is LLM_MODEL -> LLM_FALLBACK_MODELS -> LLM_PAID_FALLBACK_MODELS. Each call uses the
    first model in the chain that isn't blocked:
      - daily free limit (free-models-per-day)  -> block ALL free models until the reset, move on
      - per-minute free limit                    -> wait for the reset, retry the same model
      - upstream rate limit / no endpoint / 5xx  -> block that model for a while, move on
      - 402 insufficient credits                 -> block all paid models, move on
    """

    TRANSIENT_STATUS = {408, 500, 502, 503, 504}

    def __init__(self, cfg: Settings, blocks: ModelBlocks = MODEL_BLOCKS):
        from openai import OpenAI

        if not cfg.openrouter_api_key:
            raise LLMError("OPENROUTER_API_KEY is not set. Get a key at https://openrouter.ai/keys and put it in .env")
        self.cfg = cfg
        self.name = cfg.llm_model
        self.usage = Usage()
        self.blocks = blocks
        self.chain = list(dict.fromkeys([cfg.llm_model, *cfg.llm_fallback_models, *cfg.llm_paid_fallback_models]))
        self.client = OpenAI(
            base_url=cfg.openrouter_base_url,
            api_key=cfg.openrouter_api_key,
            timeout=cfg.llm_timeout_seconds,
            max_retries=0,  # retries/switching are handled here
        )
        headers = {"X-Title": cfg.openrouter_app_name}
        if cfg.openrouter_site_url:
            headers["HTTP-Referer"] = cfg.openrouter_site_url
        self._headers = headers
        self._json: dict[str, bool] = {m: cfg.llm_json_mode for m in self.chain}
        self._catalog: dict[str, dict] = {}
        self._info: ModelInfo | None = None
        self._sleep = time.sleep

    # ----- model metadata -----

    def _model_info_for(self, model: str) -> ModelInfo:
        meta = self._catalog[model]
        top = meta.get("top_provider") or {}
        params = meta.get("supported_parameters") or []
        return ModelInfo(
            id=model,
            context_length=top.get("context_length") or meta.get("context_length"),
            max_completion_tokens=top.get("max_completion_tokens"),
            supports_json_mode=("response_format" in params) if params else None,
            prompt_price_per_m=_price_per_m((meta.get("pricing") or {}).get("prompt")),
            completion_price_per_m=_price_per_m((meta.get("pricing") or {}).get("completion")),
        )

    def model_info(self) -> ModelInfo:
        """Info for the primary model, with context/output limits reduced to the smallest in the
        chain so that any model the client switches to can handle the same chunks."""
        if self._info is not None:
            return self._info
        try:
            self._catalog = fetch_model_catalog(self.cfg.openrouter_base_url)
        except Exception as e:
            log.warning("Could not load OpenRouter model catalog (%s); using default sizes", e)
            self._info = ModelInfo(id=self.cfg.llm_model)
            return self._info

        for model in list(self.chain):
            if model in self._catalog:
                continue
            hints = get_close_matches(model, list(self._catalog), n=5, cutoff=0.5)
            msg = f"Model '{model}' is not on OpenRouter." + (f" Did you mean: {', '.join(hints)}?" if hints else "")
            if model == self.cfg.llm_model:
                raise LLMError(msg)
            log.warning("%s Removing it from the fallback chain.", msg)
            self.chain.remove(model)

        infos = [self._model_info_for(m) for m in self.chain]
        for info in infos:
            if info.supports_json_mode is False:
                self._json[info.id] = False
        primary = infos[0]
        contexts = [i.context_length for i in infos if i.context_length]
        outputs = [i.max_completion_tokens for i in infos if i.max_completion_tokens]
        self._info = ModelInfo(
            id=primary.id,
            context_length=min(contexts) if contexts else None,
            max_completion_tokens=min(outputs) if outputs else None,
            supports_json_mode=primary.supports_json_mode,
            prompt_price_per_m=primary.prompt_price_per_m,
            completion_price_per_m=primary.completion_price_per_m,
        )
        return self._info

    def chain_status(self) -> list[tuple[str, bool, str | None]]:
        """[(model, is_free, blocked_reason)] for display."""
        return [(m, is_free_model(m, self._catalog), self.blocks.blocked(m, is_free_model(m, self._catalog)))
                for m in self.chain]

    # ----- calls with switching -----

    def _next_model(self, waited: list[float] | None = None) -> str:
        reasons = []
        for m in self.chain:
            why = self.blocks.blocked(m, is_free_model(m, self._catalog))
            if why is None:
                return m
            reasons.append(f"{m}: {why}")
        # Everything is blocked. If a model frees up soon (short cooldown), wait for it instead of failing.
        soonest = min(self.blocks.seconds_until_available(m, is_free_model(m, self._catalog)) for m in self.chain)
        if waited is not None and soonest + sum(waited) <= self.cfg.chain_max_wait_seconds:
            log.warning("All models in the chain are busy; waiting %.0fs for the first to become available", soonest)
            waited.append(soonest)
            self._sleep(soonest + 0.5)
            return self._next_model(waited)
        hint = ""
        if all(is_free_model(m, self._catalog) for m in self.chain):
            hint = " Add a paid model to LLM_PAID_FALLBACK_MODELS (needs OpenRouter credits) to keep going when free limits are reached."
        raise LLMError("All models in the chain are unavailable: " + "; ".join(reasons) + hint)

    def _switch(self, model: str, key: str, seconds: float, reason: str) -> None:
        self.blocks.block(key, seconds, reason)
        msg = f"{model} -> next model ({reason})"
        log.warning("Switching model: %s", msg)
        self.usage.add_switch(msg)

    def _handle_error(self, model: str, e: Exception, attempt: dict) -> None:
        """Decide what to do after a failed call; raises LLMError if the error is not recoverable."""
        status, msg, wait = _error_details(e)
        free = is_free_model(model, self._catalog)

        if status == 400 and "response_format" in msg and self._json.get(model):
            log.warning("%s rejected JSON mode; retrying without it", model)
            self._json[model] = False
            return
        if status == 401:
            raise LLMError(f"OpenRouter rejected the API key (401). Check OPENROUTER_API_KEY. {msg[:200]}") from e
        if status == 429 and "per-day" in msg:
            self._switch(model, ModelBlocks.FREE, wait or _seconds_to_utc_midnight(), "daily free-model limit reached")
            return
        if status == 429 and "per-min" in msg:
            attempt["waits"] = attempt.get("waits", 0) + 1
            if attempt["waits"] <= 3:
                delay = min(self.cfg.rate_limit_max_wait_seconds, (wait or 10) + 1)
                log.info("Per-minute rate limit on %s; waiting %.0fs", model, delay)
                self._sleep(delay)
                return
            self._switch(model, ModelBlocks.FREE if free else model, 60, "per-minute limit, retries exhausted")
            return
        if status == 429:
            self._switch(model, model, self.cfg.model_cooldown_seconds, "rate-limited upstream")
            return
        if status == 402:
            self._switch(model, model if free else ModelBlocks.PAID, 600, "insufficient credits (402)")
            return
        if status == 404 or (status == 400 and "no endpoints" in msg):
            self._switch(model, model, 600, "no available endpoint (check https://openrouter.ai/settings/privacy)")
            return
        if status in self.TRANSIENT_STATUS or status is None:
            attempt["transient"] = attempt.get("transient", 0) + 1
            if attempt["transient"] <= 1:
                self._sleep(2)
                return
            self._switch(model, model, self.cfg.model_cooldown_seconds, f"provider error {status or 'connection'}")
            return
        raise LLMError(f"OpenRouter call to {model} failed ({status}): {msg[:300]}") from e

    def _complete(self, messages: list[dict], max_tokens: int) -> str:
        attempts: dict[str, dict] = {}
        waited: list[float] = []
        for _ in range(4 * len(self.chain) + 4):
            model = self._next_model(waited)
            attempt = attempts.setdefault(model, {})
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": self.cfg.llm_temperature,
                "max_tokens": max_tokens,
                "extra_headers": self._headers,
                "extra_body": {"usage": {"include": True}},
            }
            if self._json.get(model):
                kwargs["response_format"] = {"type": "json_object"}
            try:
                resp = self.client.chat.completions.create(**kwargs)
            except Exception as e:
                self._handle_error(model, e, attempt)
                continue

            if not resp.choices:
                err = getattr(resp, "error", None) or (resp.model_extra or {}).get("error")
                self._switch(model, model, self.cfg.model_cooldown_seconds, f"empty response: {str(err)[:120]}")
                continue
            if resp.usage:
                extra = resp.usage.model_extra or {}
                self.usage.add(resp.model or model, resp.usage.prompt_tokens or 0,
                               resp.usage.completion_tokens or 0, float(extra.get("cost") or 0))
            choice = resp.choices[0]
            if choice.finish_reason == "length":
                raise LLMError(f"Reply from {model} was cut off at max_tokens={max_tokens}. Increase "
                               f"LLM_MAX_OUTPUT_TOKENS or decrease CHUNK_TOKENS so each chunk yields fewer risks.")
            return choice.message.content or ""
        raise LLMError("Gave up after repeated failures across the model chain")

    def chat_json(self, task: str, system: str, user: str, max_tokens: int) -> dict:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        reply = self._complete(messages, max_tokens)
        try:
            return extract_json(reply)
        except (LLMError, json.JSONDecodeError) as first_error:
            log.info("Invalid JSON from model, asking it to repair (%s)", first_error)
            messages += [
                {"role": "assistant", "content": reply},
                {"role": "user", "content": "That was not valid JSON. Reply again with ONLY the JSON object, no prose, no code fences."},
            ]
            try:
                return extract_json(self._complete(messages, max_tokens))
            except (LLMError, json.JSONDecodeError) as e:
                raise LLMError(f"Model did not return valid JSON after repair attempt: {e}") from e

    def check_key(self) -> dict:
        """Validate the API key; returns OpenRouter's key info (usage, limit)."""
        return _get_json(f"{self.cfg.openrouter_base_url.rstrip('/')}/key",
                         {"Authorization": f"Bearer {self.cfg.openrouter_api_key}"}).get("data", {})


class MockLLM:
    """Keyword rules that imitate the analyzer. For pipeline testing only."""

    name = "mock"

    RULES = [
        (r"legacy|mainframe|as400|cobol", "Integration & Dependencies", "Dependency on legacy system",
         "Integration with a legacy platform may constrain interface options, throughput and testing.", 4, 4),
        (r"third[- ]party|external api|vendor|payment gateway", "Integration & Dependencies", "Third-party service dependency",
         "Availability, rate limits and contract changes of the external provider are outside project control.", 3, 4),
        (r"\bpii\b|personal data|gdpr|pci|hipaa|card ?holder", "Security & Compliance", "Sensitive data handling and compliance",
         "Processing regulated data requires encryption, access control, auditing and compliance sign-off.", 3, 5),
        (r"99\.9+ ?%|high availability|24x7|24/7|zero downtime", "Performance & Availability", "Aggressive availability target",
         "The stated availability target requires redundancy, failover and operational maturity that may not be planned.", 3, 4),
        (r"\d[\d,]* ?(concurrent|users|transactions per|tps|requests per)", "Architecture & Scalability", "High load / scalability requirement",
         "Stated volumes require load testing and a horizontally scalable design.", 3, 4),
        (r"migrat", "Data", "Data migration risk",
         "Migrating existing data risks quality issues, downtime and reconciliation gaps.", 4, 4),
        (r"\bTBD\b|to be decided|to be confirmed|\bTBC\b", "Requirements Quality", "Unresolved requirement (TBD)",
         "Open items in the specification may cause rework or scope change late in delivery.", 4, 3),
        (r"real[- ]time", "Architecture & Scalability", "Real-time processing expectation",
         "Real-time behaviour is not quantified; latency targets and architecture impact are unclear.", 3, 3),
        (r"\bai\b|machine learning|\bml\b|blockchain", "Technology Stack & Skills", "Emerging technology / skills gap",
         "Use of specialised technology may require skills the team lacks and has uncertain maturity.", 3, 3),
    ]

    def __init__(self):
        self.usage = Usage()

    def model_info(self) -> ModelInfo:
        return ModelInfo(id="mock", context_length=DEFAULT_CONTEXT_TOKENS, max_completion_tokens=DEFAULT_MAX_OUTPUT_TOKENS)

    def chat_json(self, task: str, system: str, user: str, max_tokens: int) -> dict:
        self.usage.add("mock", len(user) // CHARS_PER_TOKEN, 0, 0.0)
        text = user.split("<document>", 1)[-1]
        if task == "profile":
            first = next((ln.strip("# ").strip() for ln in text.splitlines() if ln.strip()), "Project")
            return {"project_name": first[:80], "domain": "Unknown (mock)", "summary": "Mock profile.", "key_systems": []}
        risks = []
        for pattern, category, title, desc, lik, imp in self.RULES:
            m = re.search(pattern, text, flags=re.IGNORECASE)
            if not m:
                continue
            line = next((ln for ln in text.splitlines() if m.group(0) in ln), m.group(0)).strip()
            risks.append({
                "title": title, "category": category, "description": desc,
                "evidence": line[:300], "source_section": "", "likelihood": lik, "impact": imp,
                "mitigation": "Review with architecture team and add explicit acceptance criteria.",
                "open_questions": ["Has this been validated with the solution architect?"],
                "affected_requirements": re.findall(r"\b(?:FR|NFR|BR|REQ)[-_ ]?\d+\b", line),
            })
        return {"risks": risks}


def build_llm(cfg: Settings) -> LLM:
    if cfg.llm_provider == "mock":
        return MockLLM()
    if cfg.llm_provider == "openrouter":
        return OpenRouterLLM(cfg)
    raise ValueError(f"Unknown LLM_PROVIDER '{cfg.llm_provider}' (use 'openrouter' or 'mock')")
