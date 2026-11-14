import importlib
import math
import random
import site
import sys
from collections import Counter
from pathlib import Path

import pytest

try:
    from unit_2 import config
except ImportError as exc:
    raise RuntimeError(
        "Missing unit_2/config.py. Copy unit_2/config.py.example to unit_2/config.py "
        "and fill in your project path and module name."
    ) from exc


def _activate_project_env() -> None:
    project_root = Path(config.PROJECT_ROOT).expanduser().resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    venv_candidates: list[Path] = []
    if getattr(config, "PROJECT_VENV", None):
        venv_candidates.append(Path(config.PROJECT_VENV).expanduser().resolve())
    venv_candidates.extend([project_root / ".venv", project_root / "venv"])

    for venv_dir in venv_candidates:
        if not venv_dir.is_dir():
            continue
        for site_dir in [*venv_dir.glob("lib/python*/site-packages"), venv_dir / "Lib" / "site-packages"]:
            if site_dir.is_dir():
                site.addsitedir(str(site_dir))
        break


_activate_project_env()
_module = importlib.import_module(config.MODULE_NAME)
NGramLanguageModel = getattr(_module, config.CLASS_NAME)
UNK_TOKEN = _module.UNK_TOKEN

# Tokens and vocab are set directly to avoid depending on NLTK's tokenizer data.
TOKENS = "the cat sat on the mat the cat ran".split()
VOCAB = set(TOKENS) | {UNK_TOKEN}  # 6 words + <UNK> = 7


def make_model(n: int, tokens=TOKENS) -> NGramLanguageModel:
    model = NGramLanguageModel(n)
    model.train_tokens = list(tokens)
    model.vocab = set(tokens) | {UNK_TOKEN}
    model.train()
    return model


class TestTrain:
    def test_bigram_counts(self):
        model = make_model(2)

        assert model.ngram_counts[("the", "cat")] == 2
        assert model.ngram_counts[("cat", "sat")] == 1
        assert sum(model.ngram_counts.values()) == len(TOKENS) - 1

        assert model.history_counts[("the",)] == 3
        assert model.history_counts[("cat",)] == 2
        # "ran" is the last token, so it never has an observed continuation.
        assert ("ran",) not in model.history_counts

        assert model.next_word_counts[("the",)] == {"cat": 2, "mat": 1}
        assert model.next_word_counts[("cat",)] == {"sat": 1, "ran": 1}

    def test_trigram_counts(self):
        model = make_model(3)

        assert model.ngram_counts[("the", "cat", "sat")] == 1
        assert model.history_counts[("the", "cat")] == 2
        assert model.next_word_counts[("the", "cat")] == {"sat": 1, "ran": 1}
        assert model.history_counts[("on", "the")] == 1
        assert ("cat", "ran") not in model.history_counts

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_history_counts_equal_next_word_totals(self, n):
        model = make_model(n)

        for history, counter in model.next_word_counts.items():
            assert len(history) == n - 1
            assert model.history_counts[history] == sum(counter.values())
        assert sum(model.history_counts.values()) == sum(model.ngram_counts.values())

    def test_requires_prepared_data(self):
        with pytest.raises(ValueError):
            NGramLanguageModel(2).train()


class TestProb:
    def test_mle_bigram(self):
        model = make_model(2)

        assert model.prob("cat", ("the",)) == pytest.approx(2 / 3)
        assert model.prob("mat", ("the",)) == pytest.approx(1 / 3)
        assert model.prob("sat", ("cat",)) == pytest.approx(1 / 2)
        assert model.prob("ran", ("cat",)) == pytest.approx(1 / 2)

    def test_mle_trigram(self):
        model = make_model(3)

        assert model.prob("sat", ("the", "cat")) == pytest.approx(1 / 2)
        assert model.prob("mat", ("on", "the")) == pytest.approx(1.0)

    def test_mle_unseen_word_is_zero(self):
        model = make_model(2)

        assert model.prob("on", ("the",)) == 0.0
        assert model.prob("unseen", ("the",)) == 0.0

    def test_laplace_bigram(self):
        model = make_model(2)
        v = len(VOCAB)

        # (count + 1) / (history_count + |V|)
        assert model.prob("cat", ("the",), smoothing="Laplace") == pytest.approx(3 / (3 + v))
        assert model.prob("mat", ("the",), smoothing="Laplace") == pytest.approx(2 / (3 + v))
        # Unseen word after a seen history gets a small non-zero probability.
        assert model.prob("on", ("the",), smoothing="Laplace") == pytest.approx(1 / (3 + v))

    def test_laplace_unseen_history_is_uniform(self):
        model = make_model(2)

        # "ran" is only ever the last token, so it has no observed continuations.
        assert model.prob("cat", ("ran",), smoothing="Laplace") == pytest.approx(1 / len(VOCAB))
        assert model.prob("cat", ("zzz",), smoothing="Laplace") == pytest.approx(1 / len(VOCAB))

    @pytest.mark.parametrize("smoothing", [None, "Laplace"])
    @pytest.mark.parametrize("history", [("the",), ("cat",)])
    def test_distribution_sums_to_one(self, smoothing, history):
        model = make_model(2)

        total = sum(model.prob(w, history, smoothing=smoothing) for w in model.vocab)
        assert total == pytest.approx(1.0)

    def test_laplace_distribution_sums_to_one_for_unseen_history(self):
        model = make_model(2)

        total = sum(model.prob(w, ("ran",), smoothing="Laplace") for w in model.vocab)
        assert total == pytest.approx(1.0)

    def test_history_can_be_a_list(self):
        model = make_model(2)

        # The type hint is Sequence[str], so a list must work as well as a tuple.
        assert model.prob("cat", ["the"]) == pytest.approx(2 / 3)
        assert model.prob("cat", ["the"], smoothing="Laplace") == pytest.approx(
            model.prob("cat", ("the",), smoothing="Laplace")
        )

    def test_does_not_modify_counts(self):
        model = make_model(2)
        before = len(model.next_word_counts)

        model.prob("cat", ("ran",), smoothing="Laplace")
        model.prob("cat", ("zzz",), smoothing="Laplace")

        assert len(model.next_word_counts) == before


class TestPerplexity:
    def test_mle_bigram(self):
        model = make_model(2)

        # Predictions: P(cat|the)=2/3, P(sat|cat)=1/2 -> (1/3)^(-1/2) = sqrt(3)
        assert model.perplexity(["the", "cat", "sat"]) == pytest.approx(math.sqrt(3))

    def test_single_prediction(self):
        model = make_model(2)

        assert model.perplexity(["the", "cat"]) == pytest.approx(1.5)

    def test_mle_trigram_uses_full_history(self):
        model = make_model(3)

        # Only one prediction: P(sat | the cat) = 1/2
        assert model.perplexity(["the", "cat", "sat"]) == pytest.approx(2.0)

    def test_mle_zero_probability_gives_infinity(self):
        model = make_model(2)

        assert model.perplexity(["the", "on"]) == math.inf
        # A single zero anywhere makes the whole perplexity infinite.
        assert model.perplexity(["the", "cat", "sat", "zzz"]) == math.inf

    def test_laplace_unseen_word_is_finite(self):
        model = make_model(2)

        # P(on | the) = 1 / (3 + 7) -> perplexity 10
        assert model.perplexity(["the", "on"], smoothing="Laplace") == pytest.approx(10.0)

    def test_laplace_unseen_history_is_finite(self):
        model = make_model(2)

        # Unseen history -> uniform 1/|V| -> perplexity |V|
        assert model.perplexity(["ran", "cat"], smoothing="Laplace") == pytest.approx(len(VOCAB))

    def test_accepts_tuple_tokens(self):
        model = make_model(2)

        assert model.perplexity(("the", "cat", "sat")) == pytest.approx(math.sqrt(3))

    def test_is_at_least_one(self):
        model = make_model(2)

        assert model.perplexity(TOKENS, smoothing="Laplace") >= 1.0

    def test_deterministic_sequence_has_perplexity_one(self):
        model = make_model(2, "a b c".split())

        assert model.perplexity(["a", "b", "c"]) == pytest.approx(1.0)


class TestGenerate:
    # Deterministic chain: a -> b -> c, and "c" has no continuation.
    CHAIN = "a b c".split()
    # Cycle with a clear most-likely continuation after "the".
    CYCLE = "the cat sat on the cat sat on the mat".split()
    # Trigram cycle where every 2-token history has exactly one continuation.
    TRI_CYCLE = "a b c d a b c d".split()

    def test_greedy_picks_most_frequent_continuation(self):
        model = make_model(2, self.CYCLE)

        # the -> cat (2 vs 1), cat -> sat, sat -> on, on -> the, the -> cat
        assert model.generate(("the",), 5, "greedy") == ["cat", "sat", "on", "the", "cat"]

    def test_greedy_is_deterministic(self):
        model = make_model(2, self.CYCLE)

        runs = [model.generate(("the",), 6, "greedy") for _ in range(5)]
        assert all(run == runs[0] for run in runs)

    def test_returns_only_generated_tokens(self):
        model = make_model(2, self.CYCLE)

        out = model.generate(("the",), 3, "greedy")
        assert out == ["cat", "sat", "on"]
        assert len(out) == 3

    def test_stops_when_history_has_no_continuation(self):
        model = make_model(2, self.CHAIN)

        assert model.generate(("a",), 10, "greedy") == ["b", "c"]

    def test_unseen_history_generates_nothing(self):
        model = make_model(2, self.CHAIN)

        assert model.generate(("zzz",), 10, "greedy") == []
        assert model.generate(("c",), 10, "greedy") == []

    def test_max_tokens_zero(self):
        model = make_model(2, self.CYCLE)

        assert model.generate(("the",), 0, "greedy") == []

    def test_never_exceeds_max_tokens(self):
        model = make_model(2, self.CYCLE)

        for strategy in ("greedy", "random", "temperature"):
            assert len(model.generate(("the",), 4, strategy)) <= 4

    def test_trigram_window_slides(self):
        model = make_model(3, self.TRI_CYCLE)

        assert model.generate(("a", "b"), 5, "greedy") == ["c", "d", "a", "b", "c"]

    def test_history_can_be_a_list_and_is_not_mutated(self):
        model = make_model(3, self.TRI_CYCLE)
        history = ["a", "b"]

        out = model.generate(history, 4, "greedy")

        assert out == ["c", "d", "a", "b"]
        assert history == ["a", "b"]

    @pytest.mark.parametrize("strategy", ["random", "temperature"])
    def test_sampled_tokens_follow_observed_transitions(self, strategy):
        model = make_model(2, TOKENS)
        random.seed(0)

        for _ in range(50):
            prev = "the"
            for token in model.generate(("the",), 6, strategy):
                assert token in model.next_word_counts[(prev,)]
                prev = token

    @pytest.mark.parametrize("strategy", ["random", "temperature"])
    def test_sampling_is_reproducible_with_seed(self, strategy):
        model = make_model(2, TOKENS)

        random.seed(123)
        first = model.generate(("the",), 8, strategy)
        random.seed(123)
        second = model.generate(("the",), 8, strategy)

        assert first == second

    def test_random_matches_mle_distribution(self):
        model = make_model(2, TOKENS)
        random.seed(0)

        samples = Counter(model.generate(("the",), 1, "random")[0] for _ in range(3000))

        # P(cat|the)=2/3, P(mat|the)=1/3
        assert samples["cat"] / 3000 == pytest.approx(2 / 3, abs=0.05)
        assert samples["mat"] / 3000 == pytest.approx(1 / 3, abs=0.05)

    def test_temperature_one_matches_mle_distribution(self):
        model = make_model(2, TOKENS)
        random.seed(0)

        samples = Counter(
            model.generate(("the",), 1, "temperature", temperature=1.0)[0] for _ in range(3000)
        )

        assert samples["cat"] / 3000 == pytest.approx(2 / 3, abs=0.05)

    def test_low_temperature_approaches_greedy(self):
        model = make_model(2, TOKENS)
        random.seed(0)

        samples = [
            model.generate(("the",), 1, "temperature", temperature=0.05)[0] for _ in range(300)
        ]

        assert all(token == "cat" for token in samples)

    def test_high_temperature_approaches_uniform(self):
        model = make_model(2, TOKENS)
        random.seed(0)

        samples = Counter(
            model.generate(("the",), 1, "temperature", temperature=100.0)[0] for _ in range(3000)
        )

        # Two candidates (cat, mat) -> roughly 50/50 instead of 2/3 vs 1/3.
        assert samples["cat"] / 3000 == pytest.approx(0.5, abs=0.05)
