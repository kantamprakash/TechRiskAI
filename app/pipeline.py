"""Agent pipeline: profile -> per-chunk risk analysis -> grounding check -> consolidation -> scoring."""
from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from difflib import SequenceMatcher
from typing import Callable, Optional

from pydantic import ValidationError

from . import prompts
from .config import Settings, settings as default_settings
from .llm import LLM, Sizing, build_llm, resolve_sizing
from .models import AnalysisResult, Chunk, ProjectProfile, Risk, RiskItem, SourceDocument
from .parsing import chunk_document
from .rag import ChunkContext, ProjectRetriever

log = logging.getLogger(__name__)

ProgressFn = Callable[[int, int, str], None]


# ---------- agent 1: project profiler ----------

def profile_project(llm: LLM, docs: list[SourceDocument], project_name: str, sizing: Sizing) -> ProjectProfile:
    primary = next((d for d in docs if d.doc_type == "BRD"), docs[0])
    try:
        data = llm.chat_json("profile", prompts.PROFILE_SYSTEM,
                             prompts.profile_user(primary.doc_type, primary.text[:min(24000, sizing.chunk_chars)]),
                             max_tokens=1000)
        profile = ProjectProfile(
            project_name=str(data.get("project_name") or ""),
            domain=str(data.get("domain") or ""),
            summary=str(data.get("summary") or ""),
            key_systems=[str(s) for s in data.get("key_systems") or [] if s],
        )
    except Exception as e:
        log.warning("Profiling failed, continuing without profile: %s", e)
        profile = ProjectProfile()
    if project_name:
        profile.project_name = project_name
    profile.project_name = profile.project_name or primary.name
    return profile


# ---------- agent 2: risk analyzer (one call per chunk) ----------

def analyze_chunk(llm: LLM, chunk: Chunk, context: str, max_tokens: int,
                  rag: Optional[ChunkContext] = None) -> list[RiskItem]:
    rag = rag or ChunkContext()
    data = llm.chat_json(
        "risks",
        prompts.RISK_SYSTEM,
        prompts.risk_user(context, chunk.document.doc_type, chunk.document.name, chunk.section, chunk.text,
                          related=rag.cross_doc, knowledge=rag.knowledge),
        max_tokens=max_tokens,
    )
    raw = data.get("risks", [])
    if isinstance(raw, dict):
        raw = [raw]
    items = []
    for r in raw if isinstance(raw, list) else []:
        if not isinstance(r, dict) or not str(r.get("title", "")).strip():
            continue
        try:
            items.append(RiskItem(**r))
        except ValidationError as e:
            log.info("Skipping malformed risk: %s", e)
    return items


# ---------- grounding check ----------

def _norm(s: str) -> str:
    s = s.lower().replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"[^a-z0-9%.,:;/()'\-\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip(" .\"'")


def evidence_in_text(evidence: str, text: str) -> bool:
    """True if the quote (or each '...'-separated part of it) appears in the source, allowing small differences."""
    ev = _norm(evidence)
    if len(ev) < 8:
        return False
    src = _norm(text)
    parts = [p.strip() for p in re.split(r"\.\.\.|…", ev) if len(p.strip()) >= 8] or [ev]
    for part in parts:
        if part in src:
            continue
        match = SequenceMatcher(None, part, src, autojunk=False).find_longest_match(0, len(part), 0, len(src))
        if match.size < 0.8 * len(part):
            return False
    return True


# ---------- agent 3: consolidator (deterministic) ----------

def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def _merge_text(a: str, b: str, sep: str = "\n") -> str:
    if not b or _norm(b) in _norm(a):
        return a
    if not a:
        return b
    return f"{a}{sep}{b}"


def _unique(items: list[str]) -> list[str]:
    seen, out = set(), []
    for i in items:
        k = _norm(i)
        if k and k not in seen:
            seen.add(k)
            out.append(i)
    return out


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def consolidate(risks: list[Risk], threshold: float = 0.72,
                vectors: Optional[list[list[float]]] = None, semantic_threshold: float = 0.0) -> list[Risk]:
    """Merge duplicate risks. Two risks are duplicates when, in the same category, their titles are
    near-identical or they quote the same evidence; or, when embedding `vectors` are given, when the
    cosine similarity of their title+description is >= semantic_threshold (any category).
    The highest-scoring risk of each group is kept as the representative."""
    order = sorted(range(len(risks)), key=lambda i: (-risks[i].score, -risks[i].impact))
    merged: list[Risk] = []
    merged_vecs: list[Optional[list[float]]] = []
    for i in order:
        r = risks[i]
        vec = vectors[i] if vectors else None
        target = None
        for m, mvec in zip(merged, merged_vecs):
            same_cat = m.category == r.category and (
                _similar(m.title, r.title) >= threshold or (r.evidence and _norm(r.evidence) == _norm(m.evidence)))
            semantic = semantic_threshold > 0 and vec is not None and mvec is not None \
                and _cosine(vec, mvec) >= semantic_threshold
            if same_cat or semantic:
                target = m
                break
        if target is None:
            merged.append(r.model_copy(deep=True))
            merged_vecs.append(vec)
            continue
        target.occurrences += 1
        target.likelihood = max(target.likelihood, r.likelihood)
        target.impact = max(target.impact, r.impact)
        if len(r.description) > len(target.description):
            target.description = r.description
        target.mitigation = _merge_text(target.mitigation, r.mitigation)
        if target.evidence.count("\n") < 2:
            target.evidence = _merge_text(target.evidence, r.evidence, "\n---\n")
        target.source_section = _merge_text(target.source_section, r.source_section, "; ")
        target.source_document = "; ".join(_unique(target.source_document.split("; ") + r.source_document.split("; ")))
        target.evidence_verified = target.evidence_verified or r.evidence_verified
        target.open_questions = _unique(target.open_questions + r.open_questions)
        target.affected_requirements = _unique(target.affected_requirements + r.affected_requirements)
        target.reference = _merge_text(target.reference, r.reference, "; ")
    return merged


# ---------- orchestrator ----------

def run_analysis(
    docs: list[SourceDocument],
    project_name: str = "",
    cfg: Settings = default_settings,
    llm: Optional[LLM] = None,
    progress: Optional[ProgressFn] = None,
    retriever: Optional[ProjectRetriever] = None,
) -> AnalysisResult:
    if not docs:
        raise ValueError("At least one document (BRD or SRD) is required")
    started = datetime.now()
    llm = llm or build_llm(cfg)
    report = progress or (lambda done, total, msg: log.info("[%s/%s] %s", done, total, msg))

    info = llm.model_info()
    sizing = resolve_sizing(cfg, info)
    log.info("Model %s: context=%s tokens, max output=%s, chunk=%s tokens",
             info.id, sizing.context_tokens, sizing.max_output_tokens, sizing.chunk_tokens)

    chunks = [c for d in docs for c in chunk_document(d, sizing.chunk_chars, sizing.overlap_chars)]
    total = len(chunks) + 1

    report(0, total, f"Profiling project with {info.id}")
    profile = profile_project(llm, docs, project_name, sizing)
    context = f"{profile.project_name}. Domain: {profile.domain}. {profile.summary}".strip()

    # RAG: index the documents (+ knowledge base) and retrieve related context for every chunk up front.
    contexts: dict[int, ChunkContext] = {}
    rag_status, rag_stats = "disabled", {}
    if cfg.rag_enabled:
        report(1, total, f"Indexing documents for retrieval ({cfg.embedding_model})")
        try:
            retriever = retriever or ProjectRetriever(cfg)
            retriever.index(chunks)
            contexts = {id(c): retriever.context_for(c) for c in chunks}
            kb = retriever.kb_sync
            rag_status = "enabled"
            rag_stats = {
                "rag_pieces_indexed": retriever.pieces_indexed,
                "knowledge_base_files": kb.files if kb else 0,
            }
        except Exception as e:
            log.warning("RAG failed, continuing without retrieval: %s", e)
            rag_status = f"failed – analysed without retrieval ({str(e)[:200]})"
        finally:
            if retriever is not None:
                retriever.close()

    raw: list[Risk] = []
    failed: list[str] = []
    done = 1
    report(done, total, f"Analyzing {len(chunks)} chunk(s)")

    doc_label = {d.name: f"{d.name} ({d.doc_type})" for d in docs}

    def work(chunk: Chunk) -> list[Risk]:
        out = []
        rag = contexts.get(id(chunk))
        for item in analyze_chunk(llm, chunk, context, sizing.max_output_tokens, rag):
            risk = Risk(**item.model_dump(), source_document=doc_label[chunk.document.name])
            risk.source_section = risk.source_section or chunk.section
            risk.evidence_verified = evidence_in_text(risk.evidence, chunk.text)
            if not risk.evidence_verified and rag:
                # Evidence quoted from a related excerpt of the other document (BRD<->SRD conflict / gap)
                hit = next((r for r in rag.cross_doc if evidence_in_text(risk.evidence, r.text)), None)
                if hit:
                    risk.evidence_verified = True
                    risk.source_document = f"{risk.source_document}; {doc_label.get(hit.source, hit.source)}"
            out.append(risk)
        return out

    with ThreadPoolExecutor(max_workers=cfg.llm_concurrency) as pool:
        futures = {pool.submit(work, c): c for c in chunks}
        for fut in as_completed(futures):
            c = futures[fut]
            label = f"{c.document.name} chunk {c.index + 1} [{c.section[:60]}]"
            try:
                raw.extend(fut.result())
            except Exception as e:
                log.error("Chunk failed: %s: %s", label, e)
                failed.append(f"{label}: {e}")
            done += 1
            report(done, total, f"Analyzed {label}")

    if cfg.drop_unverified_risks:
        raw = [r for r in raw if r.evidence_verified]

    vectors = None
    if rag_status == "enabled" and cfg.dedup_similarity > 0 and cfg.embedding_provider != "mock" and len(raw) > 1:
        try:
            vectors = retriever.embeddings.embed_documents([f"{r.title}. {r.description}" for r in raw])
        except Exception as e:
            log.warning("Semantic de-duplication skipped: %s", e)
    risks = consolidate(raw, vectors=vectors, semantic_threshold=cfg.dedup_similarity)
    risks.sort(key=lambda r: (-r.score, -r.impact, r.category, r.title))
    for i, r in enumerate(risks, 1):
        r.risk_id = f"TR-{i:03d}"

    report(total, total, f"Done: {len(risks)} risk(s)")
    return AnalysisResult(
        project_name=profile.project_name,
        profile=profile,
        risks=risks,
        documents=docs,
        model=llm.name,
        provider=cfg.llm_provider,
        chunks_total=len(chunks),
        chunks_failed=failed,
        started_at=started.isoformat(timespec="seconds"),
        finished_at=datetime.now().isoformat(timespec="seconds"),
        context_tokens=sizing.context_tokens,
        max_output_tokens=sizing.max_output_tokens,
        chunk_tokens=sizing.chunk_tokens,
        llm_calls=llm.usage.calls,
        prompt_tokens=llm.usage.prompt_tokens,
        completion_tokens=llm.usage.completion_tokens,
        cost_usd=round(llm.usage.cost_usd, 4),
        models_used=dict(llm.usage.models_used),
        model_switches=list(llm.usage.switches),
        rag_status=rag_status,
        embedding_model=cfg.embedding_model if cfg.rag_enabled else "",
        embedding_calls=retriever.embeddings.calls if rag_status == "enabled" else 0,
        raw_risks=len(raw),
        **rag_stats,
    )
