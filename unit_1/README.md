# Intro to NLP — Comparison Tests

Shared pytest suites for comparing assignment outputs across different implementations.

## Unit 1 — BPE Tokenizer (`unit_1/`)

Tests `BPETokenizer` on synthetic samples and on `carroll-alice.txt` from the NLTK Gutenberg corpus (first/last 10 merge rules at 300 merges, tokenization of the 3 assignment sentences, and vocabulary size + total token counts at `n_merges` in `[50, 100, 200, 500, 1000]`).

### 1. Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Make sure your own tokenizer project already has its dependencies installed in its virtualenv (e.g. `.venv/` or `venv/` inside your project). The test runner automatically adds your project's virtualenv `site-packages` to `sys.path`.

### 2. Configure `unit_1/config.py`

Copy the example config and point it at your implementation:

```bash
cp unit_1/config.py.example unit_1/config.py
```

Edit `unit_1/config.py`:

- `PROJECT_ROOT`: absolute path to the repository containing your `BPETokenizer`.
- `MODULE_NAME`: dotted module path to import (relative to `PROJECT_ROOT`), e.g. `"src.bpe_tokenizer"` or `"bpe_tokenizer"`.
- `CLASS_NAME`: class name (`"BPETokenizer"`).
- `PRE_TOKENS_ATTR`: attribute name where `prepare_corpus` stores pre-token counts.
- `MERGE_RULES_ATTR`: attribute name where `train_bpe` stores the list of `(left, right)` merge rules (or `None` if `train_bpe` returns `(merge_rules, vocabulary)`).
- `VOCABULARY_ATTR`: attribute name where `train_bpe` stores the vocabulary (or `None` if `train_bpe` returns `(merge_rules, vocabulary)`).
- `PROJECT_VENV`: optional path to your project's virtualenv if it is not located at `<PROJECT_ROOT>/.venv` or `<PROJECT_ROOT>/venv`.

#### Custom internal data structures

If your class stores pre-tokens, merge rules, or the vocabulary in a custom format, you can define any of these optional functions directly in `unit_1/config.py`:

```python
def extract_pretoken_counts(tokenizer) -> dict[str, int]:
    ...

def extract_merge_rules(tokenizer, train_return=None) -> list[tuple[str, str]]:
    ...

def extract_vocabulary(tokenizer, train_return=None) -> set[str]:
    ...
```

### 3. Run the tests

```bash
uv run pytest -v
```
