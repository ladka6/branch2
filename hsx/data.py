"""Fixed-length context sampling from a few corpora."""

import random
from typing import Any, Dict, List, Optional

import torch

CORPUS_PRESETS = {
    "wikitext2": ("Salesforce/wikitext", "wikitext-2-raw-v1", "validation", "text"),
    "humaneval": ("openai/openai_humaneval", None, "test", "prompt"),
    "ultrachat": ("HuggingFaceH4/ultrachat_200k", None, "test_sft", "messages"),
    "gsm8k": ("openai/gsm8k", "main", "test", "question"),
}


def _record_to_text(record: Dict[str, Any], field: str, tokenizer) -> str:
    value = record.get(field)
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value and isinstance(value[0], dict):
        try:
            return tokenizer.apply_chat_template(value, tokenize=False, add_generation_prompt=False)
        except Exception:
            return "\n".join(f"{m.get('role', '')}: {m.get('content', '')}" for m in value)
    return str(value)


def build_contexts(
    tokenizer,
    n_contexts: int,
    context_tokens: int,
    seed: int,
    corpus: str = "wikitext2",
    max_records: Optional[int] = 20000,
) -> torch.Tensor:
    """Return a [n_contexts, context_tokens] LongTensor of random corpus windows."""
    from datasets import load_dataset

    name, config, split, field = CORPUS_PRESETS[corpus]
    kwargs: Dict[str, Any] = {"path": name, "split": split}
    if config:
        kwargs["name"] = config
    ds = load_dataset(**kwargs)

    chunks: List[List[int]] = []
    for i, rec in enumerate(ds):
        if max_records and i >= max_records:
            break
        text = _record_to_text(rec, field, tokenizer)
        if not text or text.isspace():
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            if tokenizer.eos_token_id is not None:
                ids.append(tokenizer.eos_token_id)
            chunks.append(ids)
    if not chunks:
        raise RuntimeError("no usable text")

    rng = random.Random(seed)
    long_enough = [c for c in chunks if len(c) >= context_tokens + 1]
    out = []
    if long_enough:
        for _ in range(n_contexts):
            c = rng.choice(long_enough)
            s = rng.randint(0, len(c) - context_tokens - 1)
            out.append(c[s : s + context_tokens])
    else:
        pool = [t for c in chunks for t in c]
        for _ in range(n_contexts):
            s = rng.randint(0, len(pool) - context_tokens - 1)
            out.append(pool[s : s + context_tokens])
    return torch.tensor(out, dtype=torch.long)


def random_contexts(n_contexts: int, context_tokens: int, vocab: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (n_contexts, context_tokens), generator=g)
