"""
K3 Day 1: LLM API exploration.

This template now uses the Gemini API through the ``google-genai`` SDK.
The function names are kept stable so the existing project structure and tests
can keep working while the backend changes.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable

from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Model defaults
# ---------------------------------------------------------------------------
# Keep the legacy constant names so the rest of the project does not break.
GEMINI_MODEL = os.getenv("LAB_MODEL", "gemini-2.5-flash")
GEMINI_MINI_MODEL = os.getenv("LAB_MINI_MODEL", "gemini-2.5-flash-lite")
OPENAI_MODEL = GEMINI_MODEL
OPENAI_MINI_MODEL = GEMINI_MINI_MODEL


# ---------------------------------------------------------------------------
# Pricing estimates per 1K tokens in USD.
# These are lightweight estimates for the exercise and are not billing logic.
# ---------------------------------------------------------------------------
PRICING_PER_1K_TOKENS = {
    "gpt-4o": {"input": 0.0025, "output": 0.0100},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gemini-2.5-flash": {"input": 0.0015, "output": 0.0060},
    "gemini-2.5-flash-lite": {"input": 0.000075, "output": 0.0003},
}


def _get_client():
    from google import genai

    return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def _make_config(
    *,
    temperature: float,
    top_p: float | None = None,
    max_tokens: int | None = None,
    system_prompt: str | None = None,
):
    from google.genai import types

    kwargs: dict[str, Any] = {"temperature": temperature}
    if top_p is not None:
        kwargs["top_p"] = top_p
    if max_tokens is not None:
        kwargs["max_output_tokens"] = max_tokens
    if system_prompt is not None:
        kwargs["system_instruction"] = system_prompt
    return types.GenerateContentConfig(**kwargs)


def _response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text:
        return text

    # Defensive fallback for mocks or older response shapes.
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text:
                return part_text
    return ""


def _chunk_text(chunk: Any) -> str:
    text = getattr(chunk, "text", None)
    if isinstance(text, str):
        return text

    choices = getattr(chunk, "choices", None) or []
    if choices:
        delta = getattr(choices[0], "delta", None)
        content = getattr(delta, "content", "")
        return content if isinstance(content, str) else ""
    return ""


def _history_to_contents(history: list[dict[str, str]]) -> list[dict[str, Any]]:
    contents: list[dict[str, Any]] = []
    for message in history:
        role = message.get("role", "user")
        mapped_role = "model" if role == "assistant" else role
        contents.append(
            {
                "role": mapped_role,
                "parts": [{"text": message.get("content", "")}],
            }
        )
    return contents


# =======================================================================
# PART 1 - API basics
# =======================================================================
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Call Gemini and return (response_text, latency_seconds).
    The public function name is kept for backwards compatibility.
    """

    client = _get_client()
    start = time.perf_counter()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=_make_config(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        ),
    )
    latency = max(time.perf_counter() - start, 1e-9)
    return _response_text(response), latency


def call_openai_mini(
    prompt: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """Call the smaller default Gemini model."""

    return call_openai(
        prompt,
        model=OPENAI_MINI_MODEL,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )


def compare_models(prompt: str) -> dict:
    """Compare the two default models and estimate response cost."""

    gpt4o_response, gpt4o_latency = call_openai(prompt)
    mini_response, mini_latency = call_openai_mini(prompt)
    gpt4o_cost_estimate = estimate_cost(prompt, gpt4o_response, model=OPENAI_MODEL)[
        "total_cost"
    ]
    return {
        "gpt4o_response": gpt4o_response,
        "mini_response": mini_response,
        "gpt4o_latency": gpt4o_latency,
        "mini_latency": mini_latency,
        "gpt4o_cost_estimate": gpt4o_cost_estimate,
    }


# =======================================================================
# PART 2 - System prompt and token counting
# =======================================================================
def chat_with_system_prompt(
    system_prompt: str,
    user_prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """Call Gemini with a system instruction and a user prompt."""

    client = _get_client()
    start = time.perf_counter()
    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=_make_config(
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
        ),
    )
    latency = max(time.perf_counter() - start, 1e-9)
    return _response_text(response), latency


def count_tokens(text: str, model: str = OPENAI_MODEL) -> int:
    """Count tokens with tiktoken and fall back gracefully if needed."""

    try:
        import tiktoken

        try:
            encoding = tiktoken.encoding_for_model(model)
        except Exception:
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
            except Exception:
                return max(1, len(text) // 4)
        return len(encoding.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def estimate_cost(prompt: str, response: str, model: str = OPENAI_MODEL) -> dict:
    """Estimate input/output token usage and cost for one request."""

    input_tokens = count_tokens(prompt, model=model)
    output_tokens = count_tokens(response, model=model)
    pricing = PRICING_PER_1K_TOKENS.get(model, PRICING_PER_1K_TOKENS[OPENAI_MODEL])
    input_cost = input_tokens / 1000 * pricing["input"]
    output_cost = output_tokens / 1000 * pricing["output"]
    total_cost = input_cost + output_cost
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }


# =======================================================================
# PART 3 - Streaming and retry
# =======================================================================
def streaming_chatbot() -> None:
    """
    Simple CLI chatbot using Gemini streaming.
    """

    client = _get_client()
    history: list[dict[str, str]] = []

    while True:
        user_msg = input()
        if user_msg.strip().lower() in {"quit", "exit"}:
            break

        contents = _history_to_contents(history) + [
            {"role": "user", "parts": [{"text": user_msg}]}
        ]
        stream = client.models.generate_content_stream(
            model=OPENAI_MODEL,
            contents=contents,
            config=_make_config(temperature=0.7, top_p=0.9, max_tokens=256),
        )

        reply_parts: list[str] = []
        for chunk in stream:
            piece = _chunk_text(chunk)
            if piece:
                print(piece, end="", flush=True)
                reply_parts.append(piece)
        print()

        reply_text = "".join(reply_parts)
        history.extend(
            [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": reply_text},
            ]
        )
        history = history[-6:]


def retry_with_backoff(
    fn: Callable,
    max_retries: int = 3,
    base_delay: float = 0.1,
) -> Any:
    """Retry a callable with exponential backoff."""

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - intentional retry helper
            last_error = exc
            if attempt >= max_retries:
                raise
            time.sleep(base_delay * (2**attempt))
    if last_error is not None:
        raise last_error
    raise RuntimeError("retry_with_backoff reached an unexpected state")


# =======================================================================
# PART 4 - Mini assistant
# =======================================================================
def run_assistant(
    persona: str,
    get_input: Callable[[], str] = None,
    max_turns: int = None,
) -> dict:
    """Run a CLI assistant with a persistent persona and short history."""

    if get_input is None:
        get_input = input

    client = _get_client()
    history: list[dict[str, str]] = []
    num_turns = 0
    total_tokens = 0
    total_cost = 0.0

    while True:
        if max_turns is not None and num_turns >= max_turns:
            break

        user_msg = get_input()
        if user_msg.strip().lower() in {"quit", "exit"}:
            break

        contents = _history_to_contents(history) + [
            {"role": "user", "parts": [{"text": user_msg}]}
        ]

        stream = retry_with_backoff(
            lambda: client.models.generate_content_stream(
                model=OPENAI_MODEL,
                contents=contents,
                config=_make_config(
                    temperature=0.7,
                    top_p=0.9,
                    max_tokens=256,
                    system_prompt=persona,
                ),
            )
        )

        reply_parts: list[str] = []
        for chunk in stream:
            piece = _chunk_text(chunk)
            if piece:
                print(piece, end="", flush=True)
                reply_parts.append(piece)
        print()

        reply_text = "".join(reply_parts)
        history.extend(
            [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": reply_text},
            ]
        )
        history = history[-6:]

        total_tokens += count_tokens(user_msg, model=OPENAI_MODEL)
        total_tokens += count_tokens(reply_text, model=OPENAI_MODEL)
        total_cost += estimate_cost(user_msg, reply_text, model=OPENAI_MODEL)[
            "total_cost"
        ]
        num_turns += 1

    return {
        "num_turns": num_turns,
        "total_tokens": total_tokens,
        "total_cost": total_cost,
        "history": history,
    }


# =======================================================================
# Bonus helpers
# =======================================================================
def batch_compare(prompts: list[str]) -> list[dict]:
    """Run compare_models for every prompt and attach the original prompt."""

    results = []
    for prompt in prompts:
        result = compare_models(prompt)
        result["prompt"] = prompt
        results.append(result)
    return results


def format_comparison_table(results: list[dict]) -> str:
    """Format batch_compare results into a simple text table."""

    headers = [
        "Prompt",
        "GPT-4o Response",
        "Mini Response",
        "GPT-4o Latency",
        "Mini Latency",
    ]
    rows = [" | ".join(headers), " | ".join(["---"] * len(headers))]

    for item in results:
        def _short(text: str, limit: int = 40) -> str:
            text = text.strip()
            return text if len(text) <= limit else text[: limit - 3] + "..."

        rows.append(
            " | ".join(
                [
                    _short(item.get("prompt", "")),
                    _short(item.get("gpt4o_response", "")),
                    _short(item.get("mini_response", "")),
                    f"{item.get('gpt4o_latency', 0):.3f}s",
                    f"{item.get('mini_latency', 0):.3f}s",
                ]
            )
        )
    return "\n".join(rows)


if __name__ == "__main__":
    print("=== Model comparison ===")
    result = compare_models("Explain temperature vs top_p in one sentence.")
    for key, value in result.items():
        print(f"{key}: {value}")

    print("\n=== Assistant demo (type 'quit' to exit) ===")
    stats = run_assistant(
        persona="You are a friendly AI teaching assistant. Answer briefly in Vietnamese.",
    )
    print("\n--- Session stats ---")
    for key, value in stats.items():
        if key != "history":
            print(f"{key}: {value}")
