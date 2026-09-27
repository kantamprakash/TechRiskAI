"""Command line: python cli.py --brd BRD.pdf --srd SRD.docx -o output/report.xlsx [--model anthropic/claude-sonnet-5]"""
import argparse
import dataclasses
import logging
from pathlib import Path

from app.config import settings
from app.excel_report import build_workbook
from app.parsing import load_document
from app.pipeline import run_analysis


def main():
    ap = argparse.ArgumentParser(description="Analyse BRD/SRD documents for technical risks and write an Excel report")
    ap.add_argument("--brd", type=Path, help="Business Requirements Document (.pdf/.docx/.txt/.md)")
    ap.add_argument("--srd", type=Path, help="Software/System Requirements Document")
    ap.add_argument("--project", default="", help="Project name (optional, otherwise inferred)")
    ap.add_argument("--model", help=f"OpenRouter model id (default: {settings.llm_model})")
    ap.add_argument("--max-output-tokens", type=int, help="Max output tokens per call (default: auto)")
    ap.add_argument("--chunk-tokens", type=int, help="Document chunk size in tokens (default: auto)")
    ap.add_argument("--mock", action="store_true", help="Use the keyword mock instead of OpenRouter")
    ap.add_argument("--no-rag", action="store_true", help="Disable retrieval (BRD<->SRD cross-check and knowledge base)")
    ap.add_argument("--index-kb", action="store_true", help="Only (re)index the knowledge base folder and exit")
    ap.add_argument("-o", "--output", type=Path, default=Path("output/tech_risk_report.xlsx"))
    args = ap.parse_args()
    if not args.brd and not args.srd and not args.index_kb:
        ap.error("provide --brd and/or --srd")

    overrides = {k: v for k, v in {
        "llm_model": args.model,
        "llm_max_output_tokens": args.max_output_tokens,
        "chunk_tokens": args.chunk_tokens,
        "llm_provider": "mock" if args.mock else None,
        "embedding_provider": "mock" if args.mock else None,
    }.items() if v}
    if args.no_rag:
        overrides["rag_enabled"] = False
    cfg = dataclasses.replace(settings, **overrides)

    if args.index_kb:
        from app.rag import build_embeddings, sync_knowledge_base
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
        _, r = sync_knowledge_base(cfg, build_embeddings(cfg))
        print(f"Knowledge base '{cfg.knowledge_base_dir}': {r.files} file(s), {r.pieces} piece(s) indexed "
              f"with {cfg.embedding_model}. Added/updated: {r.added or 'none'}. Removed: {r.removed or 'none'}")
        return

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    for noisy in ("httpx", "httpx2", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    docs = [load_document(p.name, p.read_bytes(), t) for p, t in ((args.brd, "BRD"), (args.srd, "SRD")) if p]
    result = run_analysis(docs, args.project, cfg=cfg)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    build_workbook(result).save(args.output)
    print(f"\n{len(result.risks)} risk(s) found -> {args.output}")
    for r in result.risks[:10]:
        print(f"  {r.risk_id} [{r.severity:8}] {r.title}")
    print(f"RAG: {result.rag_status} | pieces indexed: {result.rag_pieces_indexed} | "
          f"embedding calls: {result.embedding_calls} | knowledge-base files: {result.knowledge_base_files}")
    print(f"Model: {result.model} | calls: {result.llm_calls} | tokens in/out: "
          f"{result.prompt_tokens:,}/{result.completion_tokens:,} | cost: ${result.cost_usd:.4f}")
    if result.chunks_failed:
        print(f"WARNING: {len(result.chunks_failed)} chunk(s) failed - see 'Run Info' sheet")


if __name__ == "__main__":
    main()
