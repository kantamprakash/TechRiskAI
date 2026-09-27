import dataclasses
from pathlib import Path

import pytest

from app.config import Settings
from app.llm import MockLLM
from app.parsing import chunk_document, load_document
from app.pipeline import run_analysis
from app.rag import HashingEmbeddings, ProjectRetriever, sync_knowledge_base

BRD = """# Payments BRD
## Redemption
BR-04: Customers can redeem points at checkout via the third-party payment gateway.
## Security
BR-09: Customer PII must be protected.
"""
SRD = """# Payments SRD
## Payment integration
SR-12: The payment gateway integration uses synchronous REST calls with a 30 second timeout and no retry.
## Reporting
SR-20: Nightly batch report of redemptions is emailed to finance.
"""


@pytest.fixture
def cfg(tmp_path):
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "integration.md").write_text("## Third-party services\nPayment gateway outages need timeouts, retries and circuit breakers.\n")
    (kb / "data.md").write_text("## Data migration\nMigrations need reconciliation and rollback plans.\n")
    return Settings(llm_provider="mock", embedding_provider="mock", rag_enabled=True,
                    vector_db_dir=str(tmp_path / "chroma"), knowledge_base_dir=str(kb),
                    rag_piece_chars=300, rag_piece_overlap_chars=0, rag_cross_doc_k=2, rag_kb_k=1)


def docs():
    return [load_document("brd.md", BRD.encode(), "BRD"), load_document("srd.md", SRD.encode(), "SRD")]


def test_knowledge_base_sync_is_incremental(cfg):
    emb = HashingEmbeddings()
    store, r = sync_knowledge_base(cfg, emb)
    assert sorted(r.added) == ["data.md", "integration.md"] and r.pieces == 2

    store, r = sync_knowledge_base(cfg, emb, store)
    assert r.added == [] and r.removed == []

    kb = Path(cfg.knowledge_base_dir)
    (kb / "data.md").write_text("## Data migration\nChanged guidance.\n")
    (kb / "integration.md").unlink()
    store, r = sync_knowledge_base(cfg, emb, store)
    assert r.added == ["data.md"] and r.removed == ["integration.md"] and r.pieces == 1


def test_cross_document_and_knowledge_base_retrieval(cfg):
    chunks = [c for d in docs() for c in chunk_document(d, 200, 0)]
    retriever = ProjectRetriever(cfg, HashingEmbeddings())
    retriever.index(chunks)
    brd_payment = next(c for c in chunks if "redeem" in c.text)
    calls_after_indexing = retriever.embeddings.calls
    ctx = retriever.context_for(brd_payment)
    retriever.close()

    assert ctx.cross_doc and all(r.source == "srd.md" for r in ctx.cross_doc)   # only the OTHER document
    assert "payment gateway integration" in ctx.cross_doc[0].text
    assert ctx.knowledge[0].source == "integration.md"
    # retrieval reuses the cached piece vectors: no extra embedding calls
    assert retriever.embeddings.calls == calls_after_indexing


class RecordingLLM(MockLLM):
    """Mock that records prompts and reports a BRD<->SRD conflict quoting the SRD."""

    def __init__(self):
        super().__init__()
        self.prompts = []

    def chat_json(self, task, system, user, max_tokens):
        self.prompts.append(user)
        if task == "risks" and "redeem points" in user.split("<document>", 1)[-1]:
            return {"risks": [{
                "title": "Payment gateway calls have no retry", "category": "Integration & Dependencies",
                "evidence": "uses synchronous REST calls with a 30 second timeout and no retry",
                "likelihood": 4, "impact": 4, "reference": "Third-party services",
            }]}
        return super().chat_json(task, system, user, max_tokens)


def test_pipeline_uses_retrieved_context_and_verifies_cross_doc_evidence(cfg):
    llm = RecordingLLM()
    cfg = dataclasses.replace(cfg, chunk_tokens=50, chunk_overlap_tokens=0)
    result = run_analysis(docs(), cfg=cfg, llm=llm, retriever=ProjectRetriever(cfg, HashingEmbeddings()))

    assert result.rag_status == "enabled" and result.knowledge_base_files == 2
    risk_prompts = [p for p in llm.prompts if "<related_excerpts>" in p]
    assert risk_prompts and any("<knowledge_base>" in p for p in risk_prompts)

    risk = next(r for r in result.risks if r.title == "Payment gateway calls have no retry")
    assert risk.evidence_verified                     # quote found in the SRD excerpt, not the BRD chunk
    assert "srd.md (SRD)" in risk.source_document
    assert risk.reference == "Third-party services"


def test_rag_failure_falls_back_to_plain_analysis(cfg):
    class Broken(HashingEmbeddings):
        def embed_documents(self, texts):
            raise RuntimeError("embedding service down")

    result = run_analysis(docs(), cfg=cfg, llm=MockLLM(), retriever=ProjectRetriever(cfg, Broken()))
    assert result.rag_status.startswith("failed") and result.risks


def test_rag_disabled(cfg):
    llm = RecordingLLM()
    result = run_analysis(docs(), cfg=dataclasses.replace(cfg, rag_enabled=False), llm=llm)
    assert result.rag_status == "disabled" and not any("<related_excerpts>" in p for p in llm.prompts)
