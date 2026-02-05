from functools import lru_cache

from transformers import AutoTokenizer


@lru_cache(maxsize=1)
def tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)