"""Exercises OpenRouterLLM against a local stub of the OpenRouter API (no API key / network needed)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from app.config import Settings
from app.llm import LLMError, ModelBlocks, OpenRouterLLM, fetch_model_catalog
from app.parsing import load_document
from app.pipeline import run_analysis

SAMPLE = Path(__file__).resolve().parents[1] / "samples" / "sample_brd.md"
PAID, FREE_A, FREE_B = "anthropic/claude-sonnet-5", "nvidia/nemotron:free", "google/gemma:free"


def _model(mid, ctx, out, price):
    return {"id": mid, "context_length": ctx, "top_provider": {"context_length": ctx, "max_completion_tokens": out},
            "pricing": {"prompt": price, "completion": price},
            "supported_parameters": ["response_format", "max_tokens", "temperature"]}


CATALOG = {"data": [_model(PAID, 1_000_000, 128_000, "0.000002"),
                    _model(FREE_A, 262_144, 65_536, "0"),
                    _model(FREE_B, 131_072, 4_096, "0")]}

RISKS = ('```json\n{"risks":[{"title":"AS400 migration","category":"data",'
         '"evidence":"migrated from the current AS400 loyalty system","likelihood":"4","impact":5,'
         '"open_questions":"Who owns data cleansing?"}]}\n```')
PROFILE = '{"project_name":"Loyalty","domain":"Retail","summary":"s","key_systems":["AS400"]}'


def rate_limit(kind):
    return 429, {"error": {"code": 429, "message": f"Rate limit exceeded: {kind}. Add 10 credits to unlock more"}}


class Stub(BaseHTTPRequestHandler):
    requests: list = []
    errors: dict = {}      # model -> list of (status, body) returned before succeeding (or forever if "always")
    invalid_first = False  # first risk reply is not JSON (exercises repair)

    def log_message(self, *a):
        pass

    def _send(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(CATALOG)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Stub.requests.append({"body": body, "headers": {k.lower(): v for k, v in self.headers.items()}})
        model = body["model"]
        queue = Stub.errors.get(model)
        if queue:
            status, err = queue[0] if queue[0] != "always" else queue[1]
            if queue[0] != "always":
                queue.pop(0)
            return self._send(err, status)
        if "describe the project" in body["messages"][0]["content"]:
            content = PROFILE
        elif Stub.invalid_first and len(body["messages"]) == 2:
            content = "Here are the risks, they matter a lot"
        else:
            content = RISKS
        self._send({
            "id": "x", "object": "chat.completion", "created": 0, "model": model,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100,
                      "cost": 0.0 if model.endswith(":free") else 0.003},
        })


@pytest.fixture
def cfg():
    srv = HTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    Stub.requests, Stub.errors, Stub.invalid_first = [], {}, False
    fetch_model_catalog.cache_clear()
    yield Settings(llm_provider="openrouter", openrouter_api_key="sk-test",
                   openrouter_base_url=f"http://127.0.0.1:{srv.server_port}",
                   llm_model=FREE_A, llm_fallback_models=[FREE_B], llm_paid_fallback_models=[PAID],
                   llm_context_tokens=0, llm_max_output_tokens=0, chunk_tokens=0)
    srv.shutdown()


def make_llm(cfg, **overrides):
    llm = OpenRouterLLM(Settings(**{**cfg.__dict__, **overrides}), blocks=ModelBlocks())
    llm._sleep = lambda s: None
    return llm


def analyze(llm):
    doc = load_document(SAMPLE.name, SAMPLE.read_bytes(), "BRD")
    return run_analysis([doc], cfg=llm.cfg, llm=llm)


def test_openrouter_end_to_end(cfg):
    Stub.invalid_first = True
    llm = make_llm(cfg, llm_model=PAID, llm_fallback_models=[], llm_paid_fallback_models=[])
    result = analyze(llm)

    r = result.risks[0]
    assert (r.title, r.category, r.likelihood, r.impact, r.evidence_verified) == ("AS400 migration", "Data", 4, 5, True)
    assert (result.context_tokens, result.max_output_tokens) == (1_000_000, 8_192)
    assert result.llm_calls == 3 and result.cost_usd == pytest.approx(0.009)

    req = Stub.requests[1]
    assert req["headers"]["authorization"] == "Bearer sk-test"
    assert req["headers"]["x-title"] == "TechRisk AI"
    assert req["body"]["response_format"] == {"type": "json_object"}
    assert req["body"]["max_tokens"] == 8_192
    assert req["body"]["usage"] == {"include": True}


def test_sizing_uses_smallest_model_in_chain(cfg):
    info = make_llm(cfg).model_info()
    assert (info.id, info.context_length, info.max_completion_tokens) == (FREE_A, 131_072, 4_096)


def test_daily_free_limit_switches_to_paid_and_skips_all_free(cfg):
    Stub.errors = {FREE_A: [rate_limit("free-models-per-day")]}
    llm = make_llm(cfg)
    result = analyze(llm)

    assert result.risks and result.models_used == {PAID: 2}
    assert "daily free-model limit" in result.model_switches[0]
    # FREE_B (also free) must not be tried after the account-wide daily limit
    assert [r["body"]["model"] for r in Stub.requests] == [FREE_A, PAID, PAID]
    assert result.cost_usd == pytest.approx(0.006)


def test_upstream_rate_limit_switches_to_next_free_model(cfg):
    Stub.errors = {FREE_A: [rate_limit("nvidia/nemotron:free is temporarily rate-limited upstream")]}
    result = analyze(make_llm(cfg))
    assert result.models_used == {FREE_B: 2} and result.cost_usd == 0


def test_per_minute_limit_waits_and_retries_same_model(cfg):
    Stub.errors = {FREE_A: [rate_limit("free-models-per-min")]}
    llm = make_llm(cfg)
    waits = []
    llm._sleep = waits.append
    result = analyze(llm)
    assert result.models_used == {FREE_A: 2} and len(waits) == 1 and not result.model_switches


def test_transient_error_retries_then_switches(cfg):
    Stub.errors = {FREE_A: ["always", (503, {"error": {"code": 503, "message": "provider overloaded"}})]}
    result = analyze(make_llm(cfg))
    assert result.models_used == {FREE_B: 2}
    assert [r["body"]["model"] for r in Stub.requests][:3] == [FREE_A, FREE_A, FREE_B]


def test_waits_when_whole_chain_is_briefly_blocked(cfg):
    busy = (429, {"error": {"code": 429, "message": "temporarily rate-limited upstream"}})
    Stub.errors = {FREE_A: [busy], FREE_B: [busy]}
    llm = make_llm(cfg, llm_paid_fallback_models=[], model_cooldown_seconds=0.2)
    slept = []
    llm._sleep = lambda s: (slept.append(s), __import__("time").sleep(s))
    assert llm.chat_json("profile", "describe the project", "x", 100)["project_name"] == "Loyalty"
    assert slept and [r["body"]["model"] for r in Stub.requests] == [FREE_A, FREE_B, FREE_A]


def test_all_free_exhausted_without_paid_fallback_gives_clear_error(cfg):
    Stub.errors = {FREE_A: [rate_limit("free-models-per-day")]}
    llm = make_llm(cfg, llm_paid_fallback_models=[])
    with pytest.raises(LLMError, match="LLM_PAID_FALLBACK_MODELS"):
        llm.chat_json("profile", "describe the project", "x", 100)


def test_no_credits_on_paid_fallback(cfg):
    Stub.errors = {FREE_A: [rate_limit("free-models-per-day")],
                   PAID: ["always", (402, {"error": {"code": 402, "message": "Insufficient credits"}})]}
    llm = make_llm(cfg)
    with pytest.raises(LLMError, match="insufficient credits"):
        llm.chat_json("profile", "describe the project", "x", 100)


def test_bad_key_is_not_retried(cfg):
    Stub.errors = {FREE_A: ["always", (401, {"error": {"code": 401, "message": "No auth credentials found"}})]}
    with pytest.raises(LLMError, match="401"):
        make_llm(cfg).chat_json("profile", "describe the project", "x", 100)
    assert len(Stub.requests) == 1


def test_unknown_primary_model_suggests_alternatives(cfg):
    llm = make_llm(cfg, llm_model="anthropic/claude-sonet-5")
    with pytest.raises(LLMError, match="Did you mean: anthropic/claude-sonnet-5"):
        llm.model_info()


def test_unknown_fallback_is_dropped(cfg):
    llm = make_llm(cfg, llm_fallback_models=["gone/model:free", FREE_B])
    llm.model_info()
    assert llm.chain == [FREE_A, FREE_B, PAID]


def test_truncated_reply_is_reported(cfg):
    llm = make_llm(cfg, llm_model=PAID, llm_fallback_models=[], llm_paid_fallback_models=[])
    orig = Stub.do_POST

    def do_post(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self._send({"id": "x", "object": "chat.completion", "created": 0, "model": PAID,
                    "choices": [{"index": 0, "finish_reason": "length",
                                 "message": {"role": "assistant", "content": '{"risks": [{"title": "AS4'}}]})
    Stub.do_POST = do_post
    try:
        result = analyze(llm)
    finally:
        Stub.do_POST = orig
    assert result.risks == [] and "LLM_MAX_OUTPUT_TOKENS" in result.chunks_failed[0]


def test_missing_key():
    with pytest.raises(LLMError, match="OPENROUTER_API_KEY"):
        OpenRouterLLM(Settings(openrouter_api_key=""))
