"""FastAPI service: upload BRD/SRD, get an Excel risk register back."""
import dataclasses
import logging
import re
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response

from .config import settings
from .excel_report import to_excel_bytes
from .llm import LLMError, build_llm, resolve_sizing
from .parsing import load_document
from .pipeline import run_analysis

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
app = FastAPI(title="TechRisk AI", version="0.2.0")

XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


@app.get("/health")
def health(model: str = ""):
    """Checks the OpenRouter key and shows the resolved context/chunk sizing for the model."""
    cfg = dataclasses.replace(settings, llm_model=model) if model else settings
    info = {"status": "ok", "provider": cfg.llm_provider, "model": cfg.llm_model}
    try:
        llm = build_llm(cfg)
        mi = llm.model_info()
        sz = resolve_sizing(cfg, mi)
        info.update(context_tokens=sz.context_tokens, max_output_tokens=sz.max_output_tokens,
                    chunk_tokens=sz.chunk_tokens, price_per_m_tokens_in_out=[mi.prompt_price_per_m, mi.completion_price_per_m])
        if hasattr(llm, "chain_status"):
            info["model_chain"] = [{"model": m, "free": f, "blocked": b} for m, f, b in llm.chain_status()]
        if hasattr(llm, "check_key"):
            key = llm.check_key()
            info["key"] = {k: key.get(k) for k in ("label", "usage", "limit", "limit_remaining", "is_free_tier")}
    except Exception as e:
        info.update(status="error", error=str(e))
    return info


@app.post("/api/analyze")
async def analyze(
    brd: Optional[UploadFile] = File(None),
    srd: Optional[UploadFile] = File(None),
    project_name: str = Form(""),
    model: str = Form("", description="OpenRouter model id, e.g. anthropic/claude-sonnet-5 (default from .env)"),
):
    docs = []
    for upload, doc_type in ((brd, "BRD"), (srd, "SRD")):
        if upload is None or not upload.filename:
            continue
        try:
            docs.append(load_document(upload.filename, await upload.read(), doc_type))
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    if not docs:
        raise HTTPException(status_code=400, detail="Upload at least one file: 'brd' and/or 'srd'")

    cfg = dataclasses.replace(settings, llm_model=model) if model else settings
    try:
        result = await run_in_threadpool(run_analysis, docs, project_name, cfg)
    except LLMError as e:
        raise HTTPException(status_code=502, detail=str(e))
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", result.project_name)[:60] or "project"
    return Response(
        content=to_excel_bytes(result),
        media_type=XLSX,
        headers={
            "Content-Disposition": f'attachment; filename="TechRisk_{safe}.xlsx"',
            "X-Risk-Count": str(len(result.risks)),
            "X-Cost-USD": str(result.cost_usd),
        },
    )
