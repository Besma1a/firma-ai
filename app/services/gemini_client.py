"""
LLM client using Groq (free + fast + multilingual via Llama 3.3).

The filename `gemini_client.py` is kept to avoid touching imports in rag.py.
This module just speaks to Groq instead of Gemini now.
"""
from typing import Iterator
from groq import Groq
from app.config import settings


_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        if not settings.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Get a free key at "
                "https://console.groq.com/keys and add it to .env"
            )
        _client = Groq(api_key=settings.groq_api_key)
    return _client


_MOCK_RESPONSES = {
    "ar": "هذه إجابة تجريبية من نظام NABTA. (الوضع التجريبي مفعل.)",
    "fr": "Ceci est une réponse de test de NABTA. (Mode mock actif.)",
    "en": "This is a stub NABTA response (mock mode is on).",
}


def _detect_lang(text: str) -> str:
    if any("؀" <= c <= "ۿ" for c in text):
        return "ar"
    if any(m in text.lower() for m in ("é", "è", "ê", "à", "ô", "ç", "ù", "œ")):
        return "fr"
    return "en"


def complete(system_prompt: str, user_message: str,
             max_tokens: int | None = None) -> str:
    if settings.mock_claude:
        return _MOCK_RESPONSES[_detect_lang(user_message)]

    response = _get_client().chat.completions.create(
        model=settings.claude_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        max_tokens=max_tokens or settings.max_tokens,
        temperature=0.4,
    )
    return response.choices[0].message.content or ""


def stream(system_prompt: str, user_message: str,
           max_tokens: int | None = None) -> Iterator[str]:
    if settings.mock_claude:
        for word in _MOCK_RESPONSES[_detect_lang(user_message)].split():
            yield word + " "
        return

    response = _get_client().chat.completions.create(
        model=settings.claude_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        max_tokens=max_tokens or settings.max_tokens,
        temperature=0.4,
        stream=True,
    )
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta