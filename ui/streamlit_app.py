"""Streamlit UI: streamlit run ui/streamlit_app.py"""
import dataclasses
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st


def _load_streamlit_secrets() -> None:
    """Expose top-level Streamlit secrets (.streamlit/secrets.toml locally, the app's Secrets settings on
    Streamlit Community Cloud) as environment variables, before app.config reads them.
    Precedence: real environment variables > Streamlit secrets > .env file."""
    try:
        for key, value in st.secrets.items():
            if isinstance(value, (str, int, float, bool)):
                os.environ.setdefault(key, str(value))
    except Exception:  # no secrets file configured
        pass


_load_streamlit_secrets()

from app.config import settings  # noqa: E402  (must be imported after secrets are loaded)
from app.excel_report import to_excel_bytes  # noqa: E402
from app.llm import LLMError, build_llm, resolve_sizing  # noqa: E402
from app.parsing import SUPPORTED_EXTENSIONS, load_document  # noqa: E402
from app.pipeline import run_analysis  # noqa: E402

SUGGESTED_MODELS = [
    "anthropic/claude-sonnet-5",
    "anthropic/claude-opus-5.5",
    "google/gemini-3.8-flash",
    "openai/gpt-5.6-terra",
    "deepseek/deepseek-v4.1-flash",
    "qwen/qwen3.8-max-0902",
]

st.set_page_config(page_title="TechRisk AI", page_icon="⚠️", layout="wide")

# ---------- sidebar: model + size configuration ----------
with st.sidebar:
    st.header("Model (OpenRouter)")
    options = list(dict.fromkeys([settings.llm_model, *SUGGESTED_MODELS]))
    choice = st.selectbox("Model", options + ["Other…"], index=0)
    model = st.text_input("Model id", value="") if choice == "Other…" else choice
    fallbacks = st.text_input("Fallback models (tried when a model is rate-limited / down)",
                              ", ".join(settings.llm_fallback_models))
    paid_fallbacks = st.text_input("Paid fallback (only after free limits are reached; costs credits)",
                                   ", ".join(settings.llm_paid_fallback_models))

    st.header("Context & size")
    st.caption("0 = auto (from the model's OpenRouter metadata)")
    max_out = st.number_input("Max output tokens per call", 0, 128_000, settings.llm_max_output_tokens, step=1024)
    chunk = st.number_input("Chunk size (tokens)", 0, 1_000_000, settings.chunk_tokens, step=2000)
    max_chunk = st.number_input("Max auto chunk size (tokens)", 2_000, 1_000_000, settings.max_chunk_tokens, step=2000)
    concurrency = st.slider("Parallel calls", 1, 16, settings.llm_concurrency)

    st.header("RAG (vector DB)")
    rag_on = st.toggle("BRD↔SRD cross-check + knowledge base", value=settings.rag_enabled)
    st.caption(f"Embeddings: `{settings.embedding_model}`  \nKnowledge base: `{settings.knowledge_base_dir}/`")
    if st.button("Re-index knowledge base", disabled=not rag_on):
        from app.rag import build_embeddings, sync_knowledge_base
        try:
            _, kb = sync_knowledge_base(settings, build_embeddings(settings))
            st.success(f"{kb.files} file(s), {kb.pieces} piece(s). Updated: {', '.join(kb.added) or 'none'}")
        except Exception as e:
            st.error(f"Indexing failed: {e}")

    cfg = dataclasses.replace(
        settings,
        llm_model=model or settings.llm_model,
        llm_fallback_models=[m.strip() for m in fallbacks.split(",") if m.strip()],
        llm_paid_fallback_models=[m.strip() for m in paid_fallbacks.split(",") if m.strip()],
        llm_max_output_tokens=int(max_out),
        chunk_tokens=int(chunk),
        max_chunk_tokens=int(max_chunk),
        llm_concurrency=concurrency,
        rag_enabled=rag_on,
    )
    try:
        llm = build_llm(cfg)
        info = llm.model_info()
        sz = resolve_sizing(cfg, info)
        st.success(
            f"Context: **{sz.context_tokens:,}** tokens  \n"
            f"Chunk: **{sz.chunk_tokens:,}** tokens  \n"
            f"Max output: **{sz.max_output_tokens:,}** tokens"
            + (f"  \nPrice: ${info.prompt_price_per_m:.2f} in / ${info.completion_price_per_m:.2f} out per M tokens"
               if info.prompt_price_per_m is not None and info.completion_price_per_m is not None else "")
        )
        if hasattr(llm, "chain_status"):
            st.markdown("**Model chain**")
            for i, (m, free, blocked) in enumerate(llm.chain_status(), 1):
                tag = "free" if free else "paid"
                st.markdown(f"{i}. `{m}` ({tag}) " + (f"⛔ {blocked}" if blocked else "✅"))
        config_ok = True
    except (LLMError, ValueError) as e:
        st.error(str(e))
        config_ok = False

# ---------- main ----------
st.title("TechRisk AI – Technical Risk Analyzer")
if settings.llm_provider == "mock":
    st.warning("Running with the MOCK LLM (keyword rules). Set LLM_PROVIDER=openrouter and OPENROUTER_API_KEY in .env.")

types = [e.lstrip(".") for e in SUPPORTED_EXTENSIONS]
col1, col2 = st.columns(2)
brd = col1.file_uploader("BRD – Business Requirements Document", type=types)
srd = col2.file_uploader("SRD – Software Requirements Document", type=types)
project = st.text_input("Project name (optional)")

if st.button("Analyze technical risks", type="primary", disabled=not (brd or srd) or not config_ok):
    try:
        docs = [load_document(f.name, f.getvalue(), t) for f, t in ((brd, "BRD"), (srd, "SRD")) if f]
    except ValueError as e:
        st.error(str(e))
        st.stop()

    bar = st.progress(0.0, text="Starting…")

    def progress(done, total, msg):
        bar.progress(min(1.0, done / max(total, 1)), text=msg)

    try:
        with st.spinner(f"Analyzing with {cfg.llm_model}…"):
            st.session_state["result"] = run_analysis(docs, project, cfg=cfg, progress=progress)
    except LLMError as e:
        st.error(str(e))
        st.stop()

result = st.session_state.get("result")
if result:
    st.subheader(f"Results – {result.project_name}")
    counts = {s: sum(r.severity == s for r in result.risks) for s in ("Critical", "High", "Medium", "Low")}
    cols = st.columns(6)
    cols[0].metric("Total risks", len(result.risks))
    for c, (sev, n) in zip(cols[1:5], counts.items()):
        c.metric(sev, n)
    cols[5].metric("Cost", f"${result.cost_usd:.4f}")
    st.caption(f"{result.llm_calls} call(s) · {result.prompt_tokens:,} input / {result.completion_tokens:,} output tokens · "
               f"models: {', '.join(result.models_used)}")
    st.caption(f"RAG: {result.rag_status}" + (f" · {result.rag_pieces_indexed} pieces indexed · "
               f"{result.knowledge_base_files} knowledge-base file(s)" if result.rag_status == "enabled" else ""))
    if result.model_switches:
        st.info("Model switches during this run:\n\n" + "\n".join(f"- {m}" for m in result.model_switches))
    if result.chunks_failed:
        st.warning(f"{len(result.chunks_failed)} chunk(s) failed to analyse – see 'Run Info' sheet.")
        with st.expander("Failures"):
            for f in result.chunks_failed:
                st.text(f)

    st.dataframe(
        pd.DataFrame([{
            "ID": r.risk_id, "Severity": r.severity, "Score": r.score, "Category": r.category,
            "Title": r.title, "Evidence verified": r.evidence_verified, "Mitigation": r.mitigation,
        } for r in result.risks]),
        use_container_width=True, hide_index=True,
    )
    st.download_button(
        "Download Excel risk register",
        data=to_excel_bytes(result),
        file_name=f"TechRisk_{result.project_name[:40].replace(' ', '_')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )
