import json
from typing import Literal

OllamaModel = Literal["llama3.2:1b", "qwen2.5:7b", "qwen3.6:27b"]


def call_ollama_json(
    prompt: str,
    model: OllamaModel | str,
    system_prompt: str | None = None,
    options: dict | None = None,
) -> dict:
    """Call Ollama chat once and parse the response as JSON."""
    try:
        import ollama
    except ImportError as exc:
        raise RuntimeError(
            "Ollama support is optional. Install it with `uv sync --extra ollama`."
        ) from exc

    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = ollama.chat(
        model=model,
        messages=messages,
        format="json",
        options=options,
    )
    text = response.message.content

    return {
        "response_body": response,
        "response_text": text,
        "parsed": json.loads(text),
    }


if __name__ == "__main__":
    prompt = (
        'Return JSON with exactly one key named "value" for a realistic first name.'
    )
    model = "qwen2.5:7b"
    options = {"temperature": 0.7}

    result = call_ollama_json(
        prompt,
        model,
        options=options,
    )
    print(result)

    system_prompt = "You make up realistic first names for people from Russia."
    result = call_ollama_json(
        prompt,
        model,
        system_prompt=system_prompt,
        options=options,
    )
    print(result)
