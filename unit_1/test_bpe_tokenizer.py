import importlib
import site
import sys
from pathlib import Path
from typing import Any

import nltk
import pytest
from nltk.corpus import gutenberg

try:
    from unit_1 import config
except ImportError as exc:
    raise RuntimeError(
        "Missing unit_1/config.py. Copy unit_1/config.py.example to unit_1/config.py "
        "and fill in your project paths and attribute names."
    ) from exc


def _activate_project_env() -> None:
    project_root = Path(config.PROJECT_ROOT).expanduser().resolve()
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)

    venv_candidates: list[Path] = []
    if getattr(config, "PROJECT_VENV", None):
        venv_candidates.append(Path(config.PROJECT_VENV).expanduser().resolve())
    venv_candidates.extend([project_root / ".venv", project_root / "venv"])

    for venv_dir in venv_candidates:
        if not venv_dir.is_dir():
            continue
        site_Initialize_dirs = [
            *venv_dir.glob("lib/python*/site-packages"),
            venv_dir / "Lib" / "site-packages",
        ]
        for site_dir in site_Initialize_dirs:
            if site_dir.is_dir():
                site.addsitedir(str(site_dir))
        break


def create_tokenizer():
    _activate_project_env()
    module = importlib.import_module(config.MODULE_NAME)
    tokenizer_cls = getattr(module, config.CLASS_NAME)
    return tokenizer_cls()


def extract_pretoken_counts(tokenizer: Any) -> dict[str, int]:
    """Return {pre_token: frequency} from whatever prepare_corpus stored."""
    if hasattr(config, "extract_pretoken_counts"):
        return config.extract_pretoken_counts(tokenizer)
    raw = getattr(tokenizer, config.PRE_TOKENS_ATTR)
    if isinstance(raw, dict):
        counts: dict[str, int] = {}
        for key, value in raw.items():
            if isinstance(key, tuple):
                key = "".join(key)
            if isinstance(value, int):
                counts[key] = value
            elif hasattr(value, "count"):
                counts[key] = int(value.count)
            elif isinstance(value, dict) and "count" in value:
                counts[key] = int(value["count"])
            else:
                raise TypeError(
                    f"Unsupported pre-token value {value!r}; define extract_pretoken_counts(tokenizer) in unit_1/config.py."
                )
        return counts
    raise TypeError(
        f"Unsupported {config.PRE_TOKENS_ATTR!r} container {type(raw).__name__}; define extract_pretoken_counts(tokenizer) in unit_1/config.py."
    )


def extract_merge_rules(tokenizer: Any, train_return: Any = None) -> list[tuple[str, str]]:
    """Return the learned merge rules as a list of (left, right) pairs."""
    if hasattr(config, "extract_merge_rules"):
        return config.extract_merge_rules(tokenizer, train_return)
    raw = getattr(tokenizer, config.MERGE_RULES_ATTR, None) if config.MERGE_RULES_ATTR else None
    if raw is None and train_return is not None:
        raw = train_return[0] if isinstance(train_return, tuple) else train_return
    rules: list[tuple[str, str]] = []
    for item in raw:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            rules.append((str(item[0]), str(item[1])))
        elif hasattr(item, "left") and hasattr(item, "right"):
            rules.append((str(item.left), str(item.right)))
        else:
            raise TypeError(
                f"Unsupported merge rule {item!r}; define extract_merge_rules(tokenizer, train_return) in unit_1/config.py."
            )
    return rules


def extract_vocabulary(tokenizer: Any, train_return: Any = None) -> set[str]:
    """Return the learned vocabulary as a set of token strings."""
    if hasattr(config, "extract_vocabulary"):
        return config.extract_vocabulary(tokenizer, train_return)
    raw = getattr(tokenizer, config.VOCABULARY_ATTR, None) if config.VOCABULARY_ATTR else None
    if raw is None and isinstance(train_return, tuple) and len(train_return) > 1:
        raw = train_return[1]
    if isinstance(raw, dict):
        return {str(token) for token in raw.keys()}
    return {str(token) for token in raw}


# ============================================================================
# Reference outputs
# ============================================================================
CORPUS_FILE = "carroll-alice.txt"

ASSIGNMENT_EXAMPLE_TEXT = "Alice was 7 years old, wasn't she?"
ASSIGNMENT_EXAMPLE_PRE_TOKENS = [
    "Alice", "was", "7", "years", "old", ",", "wasn", "'", "t", "she", "?",
]

SMALL_SAMPLES = [
    {
        "name": "random_string",
        "text": "Ab c8as d9800- 7as8 x8$#aasd$ A@23 $%",
        "pre_tokens": [
            "Ab", "c", "8", "as", "d", "9800", "-", "7", "as", "8",
            "x", "8", "$", "#", "aasd", "$", "A", "@", "23", "$", "%",
        ],
        "counts": {
            "Ab": 1, "c": 1, "8": 3, "as": 2, "d": 1, "9800": 1, "-": 1,
            "7": 1, "x": 1, "$": 3, "#": 1, "aasd": 1, "A": 1, "@": 1,
            "23": 1, "%": 1,
        },
        "n_merges": 7,
        "merge_rules": [
            ("a", "s"),
            ("A", "b"),
            ("9", "8"),
            ("98", "0"),
            ("980", "0"),
            ("a", "as"),
            ("aas", "d"),
        ],
        "vocabulary": {
            "#", "$", "%", "-", "0", "2", "3", "7", "8", "9", "98", "980", "9800",
            "@", "A", "Ab", "a", "aas", "aasd", "as", "b", "c", "d", "s", "x",
        },
    },
    {
        "name": "alice_opening",
        "text": "Alice was beginning to get very tired",
        "pre_tokens": ["Alice", "was", "beginning", "to", "get", "very", "tired"],
        "counts": {
            "Alice": 1, "was": 1, "beginning": 1, "to": 1,
            "get": 1, "very": 1, "tired": 1,
        },
        "n_merges": 7,
        "merge_rules": [
            ("i", "n"),
            ("A", "l"),
            ("Al", "i"),
            ("Ali", "c"),
            ("Alic", "e"),
            ("w", "a"),
            ("wa", "s"),
        ],
        "vocabulary": {
            "A", "Al", "Ali", "Alic", "Alice", "a", "b", "c", "d", "e", "g", "i",
            "in", "l", "n", "o", "r", "s", "t", "v", "w", "wa", "was", "y",
        },
    },
]

ALICE_300_FIRST_10_MERGES = [
    ("h", "e"),
    ("t", "he"),
    ("i", "n"),
    ("a", "n"),
    ("o", "u"),
    ("i", "t"),
    ("e", "r"),
    ("a", "t"),
    ("o", "n"),
    ("in", "g"),
]

ALICE_300_LAST_10_MERGES = [
    ("H", "ow"),
    ("sh", "ould"),
    ("lo", "ng"),
    ("m", "in"),
    ("D", "or"),
    ("Dor", "mouse"),
    ("E", "R"),
    ("d", "ay"),
    ("af", "ter"),
    ("O", "U"),
]

QUESTION_6_EXPECTED_TOKENS = {
    "Alice was beginning to get very tired.": [
        "Alice", "was", "beg", "in", "ning", "to", "get", "very", "ti", "re", "d", ".",
    ],
    "The rabbit disappeared down the hole.": [
        "The", "r", "ab", "bit", "d", "i", "sa", "ppe", "a", "re", "d",
        "d", "ow", "n", "the", "ho", "le", ".",
    ],
    "Modern language models are based on the Transformer architecture.": [
        "M", "o", "d", "er", "n", "l", "an", "g", "u", "a", "ge",
        "m", "o", "de", "l", "s", "a", "re", "b", "as", "ed", "on",
        "the", "T", "r", "an", "s", "for", "m", "er", "arch", "ite",
        "ct", "u", "re", ".",
    ],
}

QUESTION_7_EXPECTED_STATS = {
    50: {"vocab_size": 121, "total_tokens": 84},
    100: {"vocab_size": 171, "total_tokens": 77},
    200: {"vocab_size": 271, "total_tokens": 69},
    500: {"vocab_size": 571, "total_tokens": 61},
    1000: {"vocab_size": 1071, "total_tokens": 57},
}


# ============================================================================
# Tests
# ============================================================================
@pytest.fixture(scope="session")
def alice_corpus_text() -> str:
    nltk.download("gutenberg", quiet=True)
    return gutenberg.raw(CORPUS_FILE)


@pytest.fixture(scope="session")
def alice_tokenizer_300(alice_corpus_text: str):
    tokenizer = create_tokenizer()
    tokenizer.prepare_corpus(alice_corpus_text)
    train_return = tokenizer.train_bpe(300)
    return tokenizer, train_return


@pytest.fixture(params=SMALL_SAMPLES, ids=[s["name"] for s in SMALL_SAMPLES])
def small_sample(request):
    return request.param


def test_pre_tokenize_assignment_example():
    tokenizer = create_tokenizer()
    assert tokenizer.pre_tokenize(ASSIGNMENT_EXAMPLE_TEXT) == ASSIGNMENT_EXAMPLE_PRE_TOKENS


def test_pre_tokenize_small_samples(small_sample):
    tokenizer = create_tokenizer()
    assert tokenizer.pre_tokenize(small_sample["text"]) == small_sample["pre_tokens"]


def test_prepare_corpus_counts(small_sample):
    tokenizer = create_tokenizer()
    tokenizer.prepare_corpus(small_sample["text"])
    assert extract_pretoken_counts(tokenizer) == small_sample["counts"]


def test_train_bpe_small_samples(small_sample):
    tokenizer = create_tokenizer()
    tokenizer.prepare_corpus(small_sample["text"])
    train_return = tokenizer.train_bpe(small_sample["n_merges"])
    assert extract_merge_rules(tokenizer, train_return) == small_sample["merge_rules"]
    assert extract_vocabulary(tokenizer, train_return) == small_sample["vocabulary"]


def test_tokenize_merges_all_occurrences_in_pretoken():
    tokenizer = create_tokenizer()
    tokenizer.prepare_corpus("aaaa")
    tokenizer.train_bpe(1)
    assert tokenizer.tokenize("aaaa") == ["aa", "aa"]


def test_question_5e_alice_300_merges(alice_tokenizer_300):
    tokenizer, train_return = alice_tokenizer_300
    rules = extract_merge_rules(tokenizer, train_return)
    assert len(rules) == 300
    assert rules[:10] == ALICE_300_FIRST_10_MERGES
    assert rules[-10:] == ALICE_300_LAST_10_MERGES


@pytest.mark.parametrize(
    ("sentence", "expected_tokens"),
    list(QUESTION_6_EXPECTED_TOKENS.items()),
)
def test_question_5f_sentences_at_300_merges(
    alice_tokenizer_300,
    sentence: str,
    expected_tokens: list[str],
):
    tokenizer, _ = alice_tokenizer_300
    assert list(tokenizer.tokenize(sentence)) == expected_tokens


@pytest.mark.parametrize(
    ("n_merges", "expected"),
    list(QUESTION_7_EXPECTED_STATS.items()),
)
def test_question_5g_vocab_size_and_total_tokens(
    alice_corpus_text: str,
    n_merges: int,
    expected: dict[str, int],
):
    tokenizer = create_tokenizer()
    tokenizer.prepare_corpus(alice_corpus_text)
    train_return = tokenizer.train_bpe(n_merges)

    vocab = extract_vocabulary(tokenizer, train_return)
    total_tokens = sum(
        len(tokenizer.tokenize(sentence))
        for sentence in QUESTION_6_EXPECTED_TOKENS
    )

    assert len(vocab) == expected["vocab_size"]
    assert total_tokens == expected["total_tokens"]
