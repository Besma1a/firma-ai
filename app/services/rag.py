"""
The RAG pipeline — the heart of NABTA's AI.

Flow:
  user question
    ──> retrieve (vectorstore.search)
    ──> build_context (concatenate chunks with provenance headers)
    ──> Gemini (gemini_client.complete or gemini_client.stream)
    ──> answer + sources

This module has zero web framework knowledge. The FastAPI router calls into
it. That separation lets you reuse the same logic from a CLI, a test, or
a different framework later.
"""
from typing import Iterator

from app.config import settings
from app.services import gemini_client, vectorstore


# THE most important string in the project.
# Every behavior of the chatbot — language, tone, factual grounding — flows
# from here. Edit carefully and version-control the changes.
SYSTEM_PROMPT_TEMPLATE = """You are NABTA, an agronomic assistant for Algerian farmers.

═══════════════════════════════════════════════
LANGUAGE RULE (apply before anything else)
═══════════════════════════════════════════════
Detect the EXACT language and script of the user's message. Reply in the same.

• Algerian Darija in Arabic letters  →  reply in Darija, Arabic script
  Example Q: "كيفاش نقاوم ذبابة الزيتون؟"
  Example A: "باش تقاوم ذبابة الزيتون، رش سبينوساد كل 14 يوم من يوليو. نصب مصائد
               ماكفيل وأجمع الزيتون اللي طاح من الأرض."

• Algerian Darija in Latin letters   →  reply in Latin Darija
  Example Q: "kifech ndir l-smad lel-batata?"
  Example A: "Dir N=150, P=80, K=200 kg/ha. Qabel ma tezra dir tout le P w le K
               w thelt N fel-ard. Baqin tlethayn N dir-hom fi marhaltin okhra."

• Modern Standard Arabic (MSA)       →  reply in MSA
• French                             →  reply in French
• English                            →  reply in English
• NEVER mix languages in one answer.

═══════════════════════════════════════════════
FACTUALITY RULES (read carefully)
═══════════════════════════════════════════════
Before responding, verify EVERY fact against the CONTEXT block below.

1. ACTIVE INGREDIENTS & DOSAGES: If a specific pesticide name, active
   ingredient, or dosage is NOT present in the CONTEXT, do NOT invent it.
   Instead say: "Consultez le centre ITDAS le plus proche" (or equivalent
   in the user's language).

2. ANTI-HALLUCINATION: Do not invent variety names, brand names, company
   names, or specific numerical values that are absent from CONTEXT.
   When unsure, omit the claim rather than guess.

3. ITDAS REFERRAL: When the CONTEXT does not cover the question, always
   refer the farmer to the nearest ITDAS (Institut Technique de
   Développement de l'Agronomie Saharienne) or local agricultural chamber.

4. PRACTICAL FORMAT:
   - Dosages in kg/ha or L/ha (or kg/tree for orchards).
   - Timing in days, weeks, or growth stage names.
   - Generic active ingredient names only — never brand/commercial names.
   - Replies ≤ 150 words unless the user explicitly asks for more.
   - Assume small Algerian farm (< 5 ha), limited budget, climate from
     Mediterranean (north) to Saharan (south).

═══════════════════════════════════════════════
CONTEXT (retrieved from NABTA knowledge base)
═══════════════════════════════════════════════
{context}
"""


def build_context(retrieved: list) -> str:
    """Format retrieved chunks for inclusion in the system prompt.

    `retrieved` is a list of (Document, score) tuples from
    vectorstore.search().
    """
    if not retrieved:
        return "(no relevant context found in the knowledge base)"
    parts = []
    for i, (doc, _score) in enumerate(retrieved, 1):
        m = doc.metadata
        header = f"[Source {i} — crop={m.get('crop_id')} topic={m.get('topic')} lang={m.get('lang')}]"
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _format_sources(retrieved: list) -> list[dict]:
    """Compact source list for the API response."""
    return [{
        "crop": doc.metadata.get("crop_id"),
        "topic": doc.metadata.get("topic"),
        "lang": doc.metadata.get("lang"),
        "score": float(score),
    } for doc, score in retrieved]


def answer(message: str,
           top_k: int | None = None,
           lang_filter: str | None = None,
           crop_filter: str | None = None) -> dict:
    """Synchronous RAG answer. Returns {answer, sources, model, mock}."""
    retrieved = vectorstore.search(
        message, k=top_k, lang_filter=lang_filter, crop_filter=crop_filter)
    system = SYSTEM_PROMPT_TEMPLATE.format(context=build_context(retrieved))
    text = gemini_client.complete(system, message)
    return {
        "answer": text,
        "sources": _format_sources(retrieved),
        "model": settings.claude_model,
        "mock": settings.mock_claude,
    }


def answer_stream(message: str,
                  top_k: int | None = None,
                  lang_filter: str | None = None,
                  crop_filter: str | None = None) -> Iterator[str]:
    """Generator yielding text deltas. Wraps in SSE 'data: ' frames upstream."""
    retrieved = vectorstore.search(
        message, k=top_k, lang_filter=lang_filter, crop_filter=crop_filter)
    system = SYSTEM_PROMPT_TEMPLATE.format(context=build_context(retrieved))
    yield from gemini_client.stream(system, message)