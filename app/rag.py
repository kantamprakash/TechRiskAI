"""RAG with LangChain + Chroma.

Two vector stores, both embedded with the same model (EMBEDDING_MODEL):

* Project store (in-memory, deleted after each run): small pieces of the uploaded BRD/SRD.
  While a BRD chunk is analysed, the most related SRD pieces are retrieved (and vice versa)
  so the model can spot BRD<->SRD conflicts and gaps.
* Knowledge base (persistent, VECTOR_DB_DIR): checklists, standards and lessons learned from
  KNOWLEDGE_BASE_DIR. Relevant entries are retrieved as guidance for each chunk.

Retrieval reuses the vectors already computed for the chunk's own pieces, so a run embeds
each text exactly once.
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings
from .models import Chunk
from .parsing import SUPPORTED_EXTENSIONS, _sections, extract_text

log = logging.getLogger(__name__)

_CLIENT_SETTINGS = chromadb.config.Settings(anonymized_telemetry=False)
_COSINE = {"hnsw:space": "cosine"}


# ---------- embeddings ----------

class HashingEmbeddings(Embeddings):
    """Deterministic bag-of-words embeddings for tests / mock mode (no API calls)."""

    def __init__(self, dims: int = 512):
        self.dims = dims

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dims
        for word in re.findall(r"[a-z0-9]{3,}", text.lower()):
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dims] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class CachedEmbeddings(Embeddings):
    """Wraps an Embeddings model so each distinct text is embedded once per process."""

    def __init__(self, base: Embeddings):
        self.base = base
        self._cache: dict[str, list[float]] = {}
        self.calls = 0
        self.texts_embedded = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        missing = list(dict.fromkeys(t for t in texts if t not in self._cache))
        if missing:
            self.calls += 1
            self.texts_embedded += len(missing)
            for text, vec in zip(missing, self.base.embed_documents(missing)):
                self._cache[text] = vec
        return [self._cache[t] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def build_embeddings(cfg: Settings) -> Embeddings:
    if cfg.embedding_provider == "mock":
        return HashingEmbeddings()
    from langchain_openai import OpenAIEmbeddings

    if not cfg.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set (needed for embeddings)")
    return OpenAIEmbeddings(
        model=cfg.embedding_model,
        base_url=cfg.openrouter_base_url,
        api_key=cfg.openrouter_api_key,
        default_headers={"X-Title": cfg.openrouter_app_name},
        check_embedding_ctx_length=False,  # send raw text; OpenRouter models don't accept tiktoken ids
        chunk_size=64,                     # texts per request
        max_retries=3,
    )


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()[:60]


def _splitter(cfg: Settings) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=cfg.rag_piece_chars,
        chunk_overlap=cfg.rag_piece_overlap_chars,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


# ---------- knowledge base (persistent) ----------

@dataclass
class KBSyncResult:
    files: int = 0
    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    pieces: int = 0


def open_knowledge_base(cfg: Settings, embeddings: Embeddings) -> Chroma:
    Path(cfg.vector_db_dir).mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=f"kb-{_slug(cfg.embedding_model)}",
        embedding_function=embeddings,
        persist_directory=cfg.vector_db_dir,
        collection_metadata=_COSINE,
        client_settings=_CLIENT_SETTINGS,
    )


def sync_knowledge_base(cfg: Settings, embeddings: Embeddings, store: Chroma | None = None) -> tuple[Chroma, KBSyncResult]:
    """Index new/changed files in KNOWLEDGE_BASE_DIR and drop deleted ones. Unchanged files are skipped."""
    store = store or open_knowledge_base(cfg, embeddings)
    result = KBSyncResult()
    kb_dir = Path(cfg.knowledge_base_dir)
    files = sorted(p for p in kb_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS) \
        if kb_dir.is_dir() else []
    result.files = len(files)

    existing = store.get(include=["metadatas"])
    indexed: dict[str, tuple[str, list[str]]] = {}
    for doc_id, meta in zip(existing["ids"], existing["metadatas"]):
        src = (meta or {}).get("source", "")
        digest, ids = indexed.get(src, ((meta or {}).get("sha256", ""), []))
        ids.append(doc_id)
        indexed[src] = (digest, ids)

    current = set()
    splitter = _splitter(cfg)
    for path in files:
        source = str(path.relative_to(kb_dir))
        current.add(source)
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        old = indexed.get(source)
        if old and old[0] == digest:
            continue
        if old:
            store.delete(ids=old[1])
        try:
            text = extract_text(path.name, data)
        except ValueError as e:
            log.warning("Skipping knowledge-base file %s: %s", source, e)
            continue
        docs = []
        for heading, body in _sections(text):
            for piece in splitter.split_text(body):
                docs.append(Document(page_content=piece, metadata={"source": source, "section": heading, "sha256": digest}))
        if docs:
            store.add_documents(docs, ids=[str(uuid.uuid4()) for _ in docs])
            result.added.append(source)
            log.info("Indexed knowledge-base file %s (%d pieces)", source, len(docs))

    for source, (_, ids) in indexed.items():
        if source not in current:
            store.delete(ids=ids)
            result.removed.append(source)
    result.pieces = len(store.get(include=[])["ids"])
    return store, result


# ---------- per-run retrieval ----------

@dataclass
class Retrieved:
    text: str
    source: str
    section: str
    score: float


@dataclass
class ChunkContext:
    cross_doc: list[Retrieved] = field(default_factory=list)
    knowledge: list[Retrieved] = field(default_factory=list)


class ProjectRetriever:
    """Indexes the uploaded documents for one run and retrieves related context per analysis chunk."""

    def __init__(self, cfg: Settings, embeddings: Embeddings | None = None, kb_store: Chroma | None = None):
        self.cfg = cfg
        self.embeddings = CachedEmbeddings(embeddings or build_embeddings(cfg))
        self.kb_store = kb_store
        self.kb_sync: KBSyncResult | None = None
        self.store = Chroma(
            collection_name=f"run-{uuid.uuid4().hex}",
            embedding_function=self.embeddings,
            collection_metadata=_COSINE,
            client_settings=_CLIENT_SETTINGS,
        )
        self._chunk_pieces: dict[tuple[str, int], list[str]] = {}
        self.pieces_indexed = 0
        self._multi_doc = False

    @staticmethod
    def _key(chunk: Chunk) -> tuple[str, int]:
        return chunk.document.name, chunk.index

    def index(self, chunks: list[Chunk]) -> None:
        splitter = _splitter(self.cfg)
        docs = []
        for chunk in chunks:
            pieces = splitter.split_text(chunk.text)
            self._chunk_pieces[self._key(chunk)] = pieces
            docs += [Document(page_content=p, metadata={
                "doc_name": chunk.document.name, "doc_type": chunk.document.doc_type,
                "section": chunk.section[:200], "chunk": chunk.index,
            }) for p in pieces]
        self._multi_doc = len({c.document.name for c in chunks}) > 1
        if self._multi_doc and docs:
            self.store.add_documents(docs, ids=[str(uuid.uuid4()) for _ in docs])
        else:
            # Single document: nothing to cross-reference, but embed pieces for knowledge-base lookups.
            self.embeddings.embed_documents([d.page_content for d in docs])
        self.pieces_indexed = len(docs)

        if self.cfg.rag_kb_k > 0:
            try:
                self.kb_store, self.kb_sync = sync_knowledge_base(self.cfg, self.embeddings, self.kb_store)
            except Exception as e:
                log.warning("Knowledge base unavailable: %s", e)
                self.kb_store = None

    def _search(self, store: Chroma, vectors: list[list[float]], k: int, filter: dict | None,
                source_key: str) -> list[Retrieved]:
        best: dict[str, Retrieved] = {}
        for vec in vectors:
            # Despite its name this returns cosine *distances* (0 = identical); convert to relevance.
            for doc, distance in store.similarity_search_by_vector_with_relevance_scores(vec, k=min(3, k), filter=filter):
                score = 1.0 - distance
                if score < self.cfg.rag_min_relevance:
                    continue
                key = doc.page_content
                if key not in best or score > best[key].score:
                    best[key] = Retrieved(doc.page_content, str(doc.metadata.get(source_key, "")),
                                          str(doc.metadata.get("section", "")), score)
        return sorted(best.values(), key=lambda r: -r.score)[:k]

    def context_for(self, chunk: Chunk) -> ChunkContext:
        pieces = self._chunk_pieces.get(self._key(chunk), [])
        if not pieces:
            return ChunkContext()
        vectors = self.embeddings.embed_documents(pieces)  # cached: no API call
        ctx = ChunkContext()
        if self._multi_doc and self.cfg.rag_cross_doc_k > 0:
            ctx.cross_doc = self._search(self.store, vectors, self.cfg.rag_cross_doc_k,
                                         {"doc_name": {"$ne": chunk.document.name}}, "doc_name")
        if self.kb_store is not None and self.cfg.rag_kb_k > 0 and self.kb_sync and self.kb_sync.pieces:
            ctx.knowledge = self._search(self.kb_store, vectors, self.cfg.rag_kb_k, None, "source")
        return ctx

    def close(self) -> None:
        try:
            self.store.delete_collection()
        except Exception as e:
            log.debug("Could not delete run collection: %s", e)
