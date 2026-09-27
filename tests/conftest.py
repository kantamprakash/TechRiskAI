"""Test environment: offline embeddings and a throw-away vector DB (set before app.config is imported)."""
import os
import tempfile

os.environ["EMBEDDING_PROVIDER"] = "mock"
os.environ["VECTOR_DB_DIR"] = tempfile.mkdtemp(prefix="techrisk-chroma-")
