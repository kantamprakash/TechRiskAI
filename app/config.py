"""Runtime settings, loaded from environment / .env."""
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

CHARS_PER_TOKEN = 4  # rough average for English prose


def _bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _int_or_auto(name: str, default: str = "auto") -> int:
    """'auto' / empty / 0 -> 0, meaning 'derive from the model's OpenRouter metadata'."""
    raw = os.getenv(name, default).strip().lower()
    return 0 if raw in ("", "auto", "0") else int(raw)


def _list(name: str) -> list[str]:
    return [m.strip() for m in os.getenv(name, "").split(",") if m.strip()]


@dataclass(frozen=True)
class Settings:
    # "openrouter" (default) or "mock" (keyword rules, for testing without an API key)
    llm_provider: str = os.getenv("LLM_PROVIDER", "openrouter")

    openrouter_api_key: str = field(default=os.getenv("OPENROUTER_API_KEY", ""), repr=False)
    openrouter_base_url: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    openrouter_app_name: str = os.getenv("OPENROUTER_APP_NAME", "TechRisk AI")
    openrouter_site_url: str = os.getenv("OPENROUTER_SITE_URL", "")

    # Model selection
    llm_model: str = os.getenv("LLM_MODEL", "anthropic/claude-sonnet-5")
    # Tried in order when the primary model is rate-limited / unavailable
    llm_fallback_models: list[str] = field(default_factory=lambda: _list("LLM_FALLBACK_MODELS"))
    # Used only after every model above is blocked (e.g. daily free limit). Costs credits.
    llm_paid_fallback_models: list[str] = field(default_factory=lambda: _list("LLM_PAID_FALLBACK_MODELS"))
    # How long a rate-limited / failing model is skipped before being tried again
    model_cooldown_seconds: int = int(os.getenv("MODEL_COOLDOWN_SECONDS", "120"))
    # Max wait for a per-minute rate limit to reset before retrying
    rate_limit_max_wait_seconds: int = int(os.getenv("RATE_LIMIT_MAX_WAIT_SECONDS", "65"))
    # When every model in the chain is blocked, wait up to this long in total for one to free up
    chain_max_wait_seconds: int = int(os.getenv("CHAIN_MAX_WAIT_SECONDS", "300"))
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    llm_timeout_seconds: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "300"))
    llm_json_mode: bool = _bool("LLM_JSON_MODE", True)

    # Context / size configuration (0 = auto from the model's OpenRouter metadata)
    llm_context_tokens: int = _int_or_auto("LLM_CONTEXT_TOKENS")
    llm_max_output_tokens: int = _int_or_auto("LLM_MAX_OUTPUT_TOKENS")
    chunk_tokens: int = _int_or_auto("CHUNK_TOKENS")
    # Upper bound for auto chunk size. Smaller chunks = more focused, more thorough analysis;
    # larger chunks = fewer calls and better cross-section reasoning.
    max_chunk_tokens: int = int(os.getenv("MAX_CHUNK_TOKENS", "30000"))
    chunk_overlap_tokens: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "300"))

    llm_concurrency: int = max(1, int(os.getenv("LLM_CONCURRENCY", "4")))
    drop_unverified_risks: bool = _bool("DROP_UNVERIFIED_RISKS", False)

    # ---- RAG (LangChain + Chroma) ----
    rag_enabled: bool = _bool("RAG_ENABLED", True)
    # "openrouter" or "mock" (hashing embeddings, for tests); defaults to the LLM provider
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", os.getenv("LLM_PROVIDER", "openrouter"))
    # Any id from https://openrouter.ai/api/v1/embeddings/models . Changing it re-indexes the knowledge base.
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "nvidia/nemotron-3-embed-1b:free")
    vector_db_dir: str = os.getenv("VECTOR_DB_DIR", "data/chroma")
    knowledge_base_dir: str = os.getenv("KNOWLEDGE_BASE_DIR", "knowledge_base")
    # Size of the small pieces that are embedded and retrieved
    rag_piece_chars: int = int(os.getenv("RAG_PIECE_CHARS", "1500"))
    rag_piece_overlap_chars: int = int(os.getenv("RAG_PIECE_OVERLAP_CHARS", "200"))
    # Pieces of the OTHER document (BRD<->SRD) added to each analysis call
    rag_cross_doc_k: int = int(os.getenv("RAG_CROSS_DOC_K", "6"))
    # Knowledge-base pieces (checklists, standards, lessons learned) added to each analysis call
    rag_kb_k: int = int(os.getenv("RAG_KB_K", "4"))
    # Ignore retrieved pieces with relevance below this (0..1)
    rag_min_relevance: float = float(os.getenv("RAG_MIN_RELEVANCE", "0.0"))
    # Merge risks whose title+description embeddings have cosine similarity >= this (0 = off).
    # 0.65 is calibrated for nvidia/nemotron-3-embed-1b; re-check if you change EMBEDDING_MODEL.
    dedup_similarity: float = float(os.getenv("DEDUP_SIMILARITY", "0.65"))

    @property
    def rag_prompt_tokens(self) -> int:
        """Token budget reserved in each analysis prompt for retrieved context."""
        if not self.rag_enabled:
            return 0
        return (self.rag_cross_doc_k + self.rag_kb_k) * self.rag_piece_chars // CHARS_PER_TOKEN


settings = Settings()
