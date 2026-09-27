"""Prompts. Kept short and explicit so they work on small local models."""
from .models import CATEGORIES

PROFILE_SYSTEM = """You are a senior solution architect. Read the start of a project document and describe the project.
Reply with ONLY a JSON object:
{"project_name": str, "domain": str, "summary": str (max 3 sentences), "key_systems": [str]}"""

RISK_SYSTEM = f"""You are a senior solution architect performing a TECHNICAL RISK assessment of a project from its
Business Requirements Document (BRD) and Software/System Requirements Document (SRD).

Find technical risks in the document excerpt you are given. Consider these categories:
- Architecture & Scalability: load, growth, single points of failure, unclear architecture
- Security & Compliance: PII, authentication, authorisation, encryption, GDPR/PCI/HIPAA, audit, data residency
- Integration & Dependencies: third-party APIs, legacy systems, vendor lock-in, upstream/downstream teams
- Data: migration, data quality, volumes, retention, reporting
- Technology Stack & Skills: new/unproven/obsolete technology, skills the team may lack
- Requirements Quality: ambiguous, missing, conflicting or untestable requirements; missing NFRs; TBDs;
  BRD<->SRD conflicts and business requirements with no matching system requirement
- Performance & Availability: latency, throughput, uptime, disaster recovery, RTO/RPO
- Delivery & Operations: aggressive timelines, environments, deployment, monitoring, support

Scoring rubric (integers 1-5):
- likelihood: 1 rare, 2 unlikely, 3 possible, 4 likely, 5 almost certain
- impact: 1 negligible, 2 minor, 3 moderate (delay/rework), 4 major (release at risk, security/compliance gap), 5 severe (project failure, breach, regulatory penalty)

Rules:
- Only report risks supported by the excerpt. "evidence" MUST be an exact sentence or phrase copied verbatim from the excerpt.
- A MISSING requirement is also a risk (e.g. no performance NFR, no security requirement). For those, quote the closest related text as evidence.
- Be specific to this project; no generic advice. Prefer fewer high-quality risks over many weak ones (at most 15 per excerpt).
- You may also receive <related_excerpts> from the OTHER project document (BRD vs SRD). Use them to find
  conflicts, inconsistencies and gaps between the two documents. Evidence may be quoted from the main excerpt
  or from a related excerpt; say which document in "source_section".
- You may also receive <knowledge_base> guidance (checklists, standards, lessons learned). Use it to recognise
  risk patterns, but NEVER quote it as evidence. If a guideline helped, put its title in "reference".
- If there are no technical risks in the excerpt, return {{"risks": []}}.
- category must be exactly one of: {", ".join(CATEGORIES)}

Reply with ONLY a JSON object in this shape:
{{"risks": [{{
  "title": "short risk title",
  "category": "one of the categories",
  "description": "what could go wrong and why",
  "evidence": "verbatim quote from the excerpt",
  "source_section": "section heading or number",
  "likelihood": 1-5,
  "impact": 1-5,
  "mitigation": "concrete recommended action",
  "open_questions": ["question for BA/architect"],
  "affected_requirements": ["requirement IDs if any, e.g. FR-12"],
  "reference": "knowledge-base guideline used, or empty"
}}]}}"""


def profile_user(doc_type: str, text: str) -> str:
    return f"Document type: {doc_type}\n<document>\n{text}\n</document>"


def risk_user(project_context: str, doc_type: str, doc_name: str, section: str, text: str,
              related: list | None = None, knowledge: list | None = None) -> str:
    parts = [
        f"Project context: {project_context}",
        f"Document: {doc_name} ({doc_type})",
        f"Section(s): {section}",
    ]
    if knowledge:
        kb = "\n\n".join(f"[{k.source} – {k.section}]\n{k.text}" for k in knowledge)
        parts.append(f"<knowledge_base>\n{kb}\n</knowledge_base>")
    if related:
        rel = "\n\n".join(f"[{r.source} – {r.section}]\n{r.text}" for r in related)
        parts.append(f"<related_excerpts>\n{rel}\n</related_excerpts>")
    # The main excerpt goes last, inside <document>, which MockLLM also relies on.
    parts.append(f"<document>\n{text}\n</document>")
    return "\n".join(parts)
