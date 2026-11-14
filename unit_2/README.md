# Unit 2 — N-Gram Language Model (`unit_2/`)

Tests an `NGramLanguageModel` (Maman 12, Question 5 starter API) on a tiny hand-made corpus, with one test class per method:

- `TestTrain`: `ngram_counts`, `history_counts` and `next_word_counts` for n = 2, 3, 4.
- `TestProb`: MLE and Laplace probabilities, unseen words and histories, list histories, distributions summing to 1.
- `TestPerplexity`: only the tokens after the first `n-1` are predicted, `math.inf` on a zero probability, Laplace stays finite.
- `TestGenerate`: greedy, random and temperature strategies, early stop on a history with no continuation, sliding history window.

The tests set `train_tokens` and `vocab` directly instead of calling `prepare_data`, so no NLTK data needs to be downloaded.

## 1. Setup

Same as unit 1: Python 3.11+ and [uv](https://docs.astral.sh/uv/), then `uv sync`. Your own project must have its dependencies installed in its virtualenv (`.venv/` or `venv/` inside the project), which is added to `sys.path` automatically.

## 2. Configure `unit_2/config.py`

```bash
cp unit_2/config.py.example unit_2/config.py
```

- `PROJECT_ROOT`: absolute path to the repository containing your n-gram model.
- `MODULE_NAME`: dotted module path to import (relative to `PROJECT_ROOT`), e.g. `"n_gram_lm"`.
- `CLASS_NAME`: class name (`"NGramLanguageModel"`).
- `PROJECT_VENV`: optional path to your project's virtualenv if it is not at `<PROJECT_ROOT>/.venv` or `<PROJECT_ROOT>/venv`.

Your module must define `UNK_TOKEN`, and your class must keep the starter's attribute names (`train_tokens`, `vocab`, `ngram_counts`, `history_counts`, `next_word_counts`) and method signatures.

## 3. Run the tests

```bash
uv run pytest unit_2 -v
```

## Assumptions the tests make

- Laplace smoothing uses `len(vocab)` (which includes `<UNK>`) in the denominator, and an unseen history under Laplace gives a uniform `1/len(vocab)`.
- `generate` and sampling use Python's global `random` module (the tests seed it), and the history may be passed as a list or a tuple.
- Greedy tie-breaking, MLE on an unseen history, unknown strategies and sequences shorter than `n` are not tested, since the assignment does not specify them.
