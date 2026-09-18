"""Shared prompt construction and truncation for training and inference.

The review text alone is truncated (never the system prompt, never the target),
the model's native chat template renders the messages, and `max_new_tokens` is
reserved inside `max_seq_length`; if the rendered prompt plus the reservation
exceeds `max_seq_length` the user text is shrunk further.

Training calls `build_training_text`, which starts from the *identical* user
truncation the inference helper produced, appends the assistant target plus EOS
and asserts the full system/user/assistant/EOS sequence fits. The target is
never right-truncated.
"""

from __future__ import annotations

from .spec import SYSTEM_PROMPT


def tokenizer_of(prompt_source):
    """Underlying tokenizer: `processor.tokenizer` when present, else the object."""
    return getattr(prompt_source, "tokenizer", prompt_source)


def _token_ids(tokenizer, text: str, max_length=None) -> list:
    kwargs = {"add_special_tokens": False}
    if max_length is not None:
        kwargs.update({"truncation": True, "max_length": max(0, int(max_length))})
    return list(tokenizer(text, **kwargs)["input_ids"])


def _decode(tokenizer, ids: list) -> str:
    return tokenizer.decode(ids, skip_special_tokens=True)


def render_chat(
    prompt_source, messages: list[dict], chat_template_kwargs: dict | None = None,
    add_generation_prompt: bool = True,
) -> str:
    kwargs = dict(chat_template_kwargs or {})
    return prompt_source.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **kwargs,
    )


def truncated_user_text(tokenizer, review: str, max_user_tokens: int) -> tuple[str, int, bool]:
    """Truncate the review alone; returns (text, tokens, was_truncated)."""
    full_tokens = len(_token_ids(tokenizer, review))
    ids = _token_ids(tokenizer, review, max_user_tokens)
    return _decode(tokenizer, ids), len(ids), full_tokens > len(ids)


def _messages(system_prompt: str, user_text: str, assistant_text: str | None) -> list[dict]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    if assistant_text is not None:
        messages.append({"role": "assistant", "content": assistant_text})
    return messages


def build_generation_input(
    prompt_source,
    review: str,
    system_prompt: str = SYSTEM_PROMPT,
    *,
    max_user_tokens: int,
    max_seq_length: int,
    max_new_tokens: int,
    chat_template_kwargs: dict | None = None,
) -> dict:
    """Rendered generation prompt with `max_new_tokens` reserved inside the window."""
    tokenizer = tokenizer_of(prompt_source)
    full_tokens = len(_token_ids(tokenizer, review))
    limit = int(max_user_tokens)
    user_ids = _token_ids(tokenizer, review, limit)

    while True:
        user_text = _decode(tokenizer, user_ids)
        prompt = render_chat(
            prompt_source,
            _messages(system_prompt, user_text, None),
            chat_template_kwargs,
            add_generation_prompt=True,
        )
        prompt_tokens = len(_token_ids(tokenizer, prompt))
        if prompt_tokens + max_new_tokens <= max_seq_length:
            break
        if not user_ids:
            raise ValueError(
                f"system prompt + reserved generation tokens ({prompt_tokens + max_new_tokens}) "
                f"exceed max_seq_length={max_seq_length} with an empty user review"
            )
        excess = prompt_tokens + max_new_tokens - max_seq_length
        new_limit = max(0, len(user_ids) - max(1, excess))
        if new_limit >= len(user_ids):  # always make progress
            new_limit = len(user_ids) - 1
        user_ids = _token_ids(tokenizer, review, new_limit)

    return {
        "prompt": prompt,
        "user_text": user_text,
        "user_tokens": len(user_ids),
        "prompt_tokens": prompt_tokens,
        "review_tokens": full_tokens,
        "truncated": full_tokens > len(user_ids),
        # The review was shrunk below min(review length, max_user_tokens) because
        # the rendered prompt plus the generation reservation did not fit.
        "reduced_below_limit": len(user_ids) < min(full_tokens, int(max_user_tokens)),
    }


def build_training_text(
    prompt_source,
    review: str,
    target_text: str,
    system_prompt: str = SYSTEM_PROMPT,
    *,
    max_user_tokens: int,
    max_seq_length: int,
    max_new_tokens: int,
    chat_template_kwargs: dict | None = None,
) -> dict:
    """Full system/user/assistant/EOS training text; target never truncated."""
    tokenizer = tokenizer_of(prompt_source)
    generation = build_generation_input(
        prompt_source,
        review,
        system_prompt,
        max_user_tokens=max_user_tokens,
        max_seq_length=max_seq_length,
        max_new_tokens=max_new_tokens,
        chat_template_kwargs=chat_template_kwargs,
    )
    user_text = generation["user_text"]
    user_tokens = generation["user_tokens"]
    target_tokens = len(_token_ids(tokenizer, target_text))
    eos_token = getattr(tokenizer, "eos_token", None)

    def render(text: str) -> tuple[str, bool]:
        rendered = render_chat(
            prompt_source,
            _messages(system_prompt, text, target_text),
            chat_template_kwargs,
            add_generation_prompt=False,
        )
        appended = False
        if eos_token and not rendered.endswith(eos_token):
            rendered = rendered + eos_token
            appended = True
        return rendered, appended

    limit = user_tokens
    while True:
        full_text, eos_appended = render(user_text)
        full_tokens = len(_token_ids(tokenizer, full_text))
        if full_tokens <= max_seq_length:
            break
        if not user_text:
            raise ValueError(
                f"system/user/target/EOS ({full_tokens} tokens) exceeds "
                f"max_seq_length={max_seq_length} even with an empty user review"
            )
        limit = max(0, limit - max(1, full_tokens - max_seq_length))
        user_text = _decode(tokenizer, _token_ids(tokenizer, review, limit))

    if target_text not in full_text:
        raise RuntimeError("assistant target missing from rendered training text")
    if len(_token_ids(tokenizer, target_text)) != target_tokens:
        raise RuntimeError("assistant target was truncated")
    if full_tokens > max_seq_length:
        raise RuntimeError(
            f"training sequence {full_tokens} exceeds max_seq_length={max_seq_length}"
        )

    return {
        "text": full_text,
        "user_text": user_text,
        "user_tokens": limit,
        "prompt_tokens": generation["prompt_tokens"],
        "target_tokens": target_tokens,
        "full_tokens": full_tokens,
        "truncated": generation["truncated"] or limit < generation["user_tokens"],
        "eos_appended": eos_appended,
        "eos_present": bool(eos_token) and full_text.endswith(eos_token),
    }
