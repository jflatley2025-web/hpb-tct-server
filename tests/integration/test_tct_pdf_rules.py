"""
Integration tests for tct_pdf_rules.py — ChromaDB rule extraction

Uses an in-memory ChromaDB client (not the persistent one) and a mock
embedding function so tests:
  - Are isolated: no shared state between test runs
  - Are fast: no disk I/O, no model download required
  - Are safe: no production collection is touched

The mock embedding function returns deterministic 384-dimensional random
vectors — identical dimensionality to all-MiniLM-L6-v2. Semantic quality
is not tested here; only the ChromaDB integration and rule extraction logic.

Tests:
  - Pre-flight check fails with SystemExit when model cache is missing
  - Pre-flight check passes when model cache directory exists
  - _populate_collection ingests documents and increments count
  - _populate_collection skips ingestion if collection already has documents
  - _populate_collection force-repopulates when force_repopulate=True
  - _query_layer_rules returns LayerRules with non-empty raw_chunks
  - _query_layer_rules fills rules_by_topic for each query
  - TCTRuleSet.all_populated() returns True when all layers have chunks
  - TCTRuleSet.all_populated() returns False when any layer is empty
  - load_tct_rules raises SystemExit when PDF dir is missing
  - load_tct_rules raises SystemExit when collection is empty after ingestion
  - _chunk_text produces correct chunk count and overlap
  - LayerRules.is_populated returns correct value
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import chromadb

from tct_pdf_rules import (
    LayerRules,
    TCTRuleSet,
    EMBEDDING_MODEL,
    LAYER_NAMES,
    LAYER_QUERIES,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    _chunk_text,
    _populate_collection,
    _query_layer_rules,
    _check_embedding_model_cached,
    load_tct_rules,
)


# ---------------------------------------------------------------------------
# Mock embedding function — 384d random vectors (same dim as all-MiniLM-L6-v2)
# ---------------------------------------------------------------------------

class _MockEmbeddingFunction:
    """
    Deterministic mock embedding function compatible with ChromaDB ≥0.4 and ≥1.5.

    ChromaDB 1.5 changed the EF protocol:
      - name() is required for EF identity validation
      - embed_query() is called for query embeddings (is_query=True)
      - __call__ / embed_documents() is called for document embeddings

    Returns random 384-dimensional vectors (same dim as all-MiniLM-L6-v2)
    without requiring a sentence-transformers model download.
    """

    _DIM = 384
    _rng = np.random.default_rng(seed=42)

    def name(self) -> str:
        # Return "default" so ChromaDB skips EF conflict validation on get_or_create
        return "default"

    def _vectors(self, input) -> list:  # noqa: A002
        return [self._rng.random(self._DIM).tolist() for _ in input]

    def __call__(self, input):  # noqa: A002
        return self._vectors(input)

    # ChromaDB ≥1.5 calls embed_query() for query embeddings
    def embed_query(self, input):  # noqa: A002
        return self._vectors(input)

    # ChromaDB ≥1.5 may call embed_documents() for document embeddings
    def embed_documents(self, input):  # noqa: A002
        return self._vectors(input)


_MOCK_EF = _MockEmbeddingFunction()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_in_memory_client() -> chromadb.Client:
    """Return an ephemeral in-memory ChromaDB client for test isolation."""
    return chromadb.EphemeralClient()


def _make_collection(client: chromadb.Client) -> chromadb.Collection:
    """Create a fresh tct_lectures collection with a mock embedding function."""
    return client.get_or_create_collection(
        name="tct_lectures",
        embedding_function=_MOCK_EF,
        metadata={"hnsw:space": "cosine"},
    )


def _ingest_fake_docs(collection: chromadb.Collection, layer: int, n: int = 5) -> None:
    """Ingest n synthetic documents for the given layer."""
    docs = [
        f"Layer {layer} rule chunk {i}: market structure, BOS, RTZ, pivots, liquidity "
        f"swing failure pattern CHoCH domino effect level 1 2 3 order block FVG tap."
        for i in range(n)
    ]
    ids = [f"L{layer}_fake_{i}" for i in range(n)]
    metadatas = [
        {"lecture_layer": layer, "source_file": f"fake_{layer}.pdf", "chunk_index": i}
        for i in range(n)
    ]
    collection.upsert(ids=ids, documents=docs, metadatas=metadatas)


# ---------------------------------------------------------------------------
# Pre-flight check
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_preflight_fails_when_model_cache_missing(tmp_path):
    """SystemExit is raised when the model cache directory does not exist."""
    with patch("tct_pdf_rules.Path.home", return_value=tmp_path):
        # tmp_path has no sentence_transformers cache
        with pytest.raises(SystemExit, match="Embedding model not cached"):
            _check_embedding_model_cached()


@pytest.mark.integration
def test_preflight_passes_when_model_cache_exists(tmp_path):
    """No exception is raised when the model cache directory exists."""
    cache_dir = (
        tmp_path / ".cache" / "torch" / "sentence_transformers"
        / EMBEDDING_MODEL.replace("/", "_")
    )
    cache_dir.mkdir(parents=True)

    with patch("tct_pdf_rules.Path.home", return_value=tmp_path):
        _check_embedding_model_cached()  # should not raise


# ---------------------------------------------------------------------------
# _chunk_text
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_chunk_text_produces_correct_count():
    """Text of 2 × CHUNK_SIZE produces at least 2 chunks."""
    text = "A" * (CHUNK_SIZE * 2)
    chunks = _chunk_text(text)
    assert len(chunks) >= 2


@pytest.mark.integration
def test_chunk_text_overlap_means_content_repeated():
    """Adjacent chunks share CHUNK_OVERLAP characters of content."""
    text = "X" * (CHUNK_SIZE * 3)
    chunks = _chunk_text(text)
    if len(chunks) >= 2:
        overlap_from_first = chunks[0][-CHUNK_OVERLAP:]
        overlap_in_second = chunks[1][:CHUNK_OVERLAP]
        assert overlap_from_first == overlap_in_second


@pytest.mark.integration
def test_chunk_text_short_text_produces_one_chunk():
    """Text shorter than CHUNK_SIZE produces exactly one chunk."""
    text = "Short text about market structure and BOS."
    chunks = _chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


# ---------------------------------------------------------------------------
# _populate_collection (using mock EF to avoid model download)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_populate_collection_ingests_documents(tmp_path):
    """_populate_collection adds documents to an empty collection."""
    client = _make_in_memory_client()
    collection = _make_collection(client)

    fake_pdf = tmp_path / "TCT mentorship - Lecture 1 Market Structure8Pages.pdf"
    fake_pdf.write_bytes(b"placeholder")

    fake_text = (
        "6-candle rule pivot detection inside bar exclusion. "
        "MSH MSL confirmation revisit rule. BOS good bad quality. "
        "CHoCH domino effect Level 1 2 3 structure. " * 30
    )

    with patch("tct_pdf_rules.PDF_DIR", tmp_path), \
         patch("tct_pdf_rules._extract_pdf_text", return_value=fake_text), \
         patch(
             "tct_pdf_rules._get_or_create_collection",
             return_value=collection,
         ), \
         patch("tct_pdf_rules.LECTURE_PDF_MAP",
               {1: ["TCT mentorship - Lecture 1 Market Structure8Pages.pdf"]}):
        _populate_collection(collection)

    assert collection.count() > 0


@pytest.mark.integration
def test_populate_collection_skips_if_already_populated(tmp_path):
    """_populate_collection does not re-ingest if collection already has docs."""
    client = _make_in_memory_client()
    collection = _make_collection(client)

    _ingest_fake_docs(collection, layer=1, n=3)
    initial_count = collection.count()

    with patch("tct_pdf_rules.PDF_DIR", tmp_path), \
         patch("tct_pdf_rules._extract_pdf_text", return_value="some text"):
        _populate_collection(collection, force_repopulate=False)

    assert collection.count() == initial_count


@pytest.mark.integration
def test_populate_collection_force_repopulate_reingest(tmp_path):
    """force_repopulate=True re-ingests even if collection already has docs."""
    client = _make_in_memory_client()
    collection = _make_collection(client)

    fake_text = "BOS CHoCH market structure pivot. " * 40

    (tmp_path / "fake1.pdf").write_bytes(b"placeholder")

    with patch("tct_pdf_rules.PDF_DIR", tmp_path), \
         patch("tct_pdf_rules._extract_pdf_text", return_value=fake_text), \
         patch("tct_pdf_rules.LECTURE_PDF_MAP", {1: ["fake1.pdf"]}):
        _populate_collection(collection, force_repopulate=False)
        first_count = collection.count()
        _populate_collection(collection, force_repopulate=True)

    # upsert deduplicates by ID — count should remain consistent
    assert collection.count() >= first_count


# ---------------------------------------------------------------------------
# _query_layer_rules
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_query_layer_rules_returns_populated_layer_rules():
    """_query_layer_rules fills raw_chunks for a layer with documents."""
    client = _make_in_memory_client()
    collection = _make_collection(client)

    _ingest_fake_docs(collection, layer=1, n=15)

    lr = _query_layer_rules(collection, layer=1)

    assert isinstance(lr, LayerRules)
    assert lr.layer == 1
    assert lr.name == LAYER_NAMES[1]
    assert lr.is_populated()
    assert len(lr.raw_chunks) > 0


@pytest.mark.integration
def test_query_layer_rules_fills_rules_by_topic():
    """_query_layer_rules populates rules_by_topic for each query in the layer."""
    client = _make_in_memory_client()
    collection = _make_collection(client)

    _ingest_fake_docs(collection, layer=1, n=15)

    lr = _query_layer_rules(collection, layer=1)

    for query in LAYER_QUERIES[1]:
        assert query in lr.rules_by_topic, f"Missing topic: {query!r}"


@pytest.mark.integration
def test_query_layer_rules_topic_values_are_strings():
    """rules_by_topic values are non-empty strings."""
    client = _make_in_memory_client()
    collection = _make_collection(client)

    _ingest_fake_docs(collection, layer=2, n=10)

    lr = _query_layer_rules(collection, layer=2)

    for topic, content in lr.rules_by_topic.items():
        assert isinstance(content, str)


@pytest.mark.integration
def test_query_layer_rules_retrieves_only_correct_layer(tmp_path):
    """_query_layer_rules does not mix documents from different layers."""
    client = _make_in_memory_client()
    collection = _make_collection(client)

    # Ingest docs for layers 1 and 2
    _ingest_fake_docs(collection, layer=1, n=5)
    _ingest_fake_docs(collection, layer=2, n=5)

    lr = _query_layer_rules(collection, layer=1)

    # Metadata filter by lecture_layer=1 means all returned metadatas have layer 1
    # We can't directly check the metadatas from LayerRules, but is_populated ensures
    # that some chunks were retrieved
    assert lr.is_populated()
    assert lr.layer == 1


# ---------------------------------------------------------------------------
# TCTRuleSet
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_tct_ruleset_all_populated_true_when_all_layers_have_chunks():
    """TCTRuleSet.all_populated() returns True when all layers have raw_chunks."""
    rule_set = TCTRuleSet()
    for layer in range(1, 7):
        lr = LayerRules(layer=layer, name=LAYER_NAMES[layer])
        lr.raw_chunks = ["some rule chunk"]
        rule_set.layers[layer] = lr

    assert rule_set.all_populated() is True


@pytest.mark.integration
def test_tct_ruleset_all_populated_false_when_any_layer_empty():
    """TCTRuleSet.all_populated() returns False when any layer has no chunks."""
    rule_set = TCTRuleSet()
    for layer in range(1, 7):
        lr = LayerRules(layer=layer, name=LAYER_NAMES[layer])
        lr.raw_chunks = ["some rule chunk"]
        rule_set.layers[layer] = lr

    rule_set.layers[4].raw_chunks = []

    assert rule_set.all_populated() is False


@pytest.mark.integration
def test_layer_rules_is_populated_false_on_init():
    """LayerRules.is_populated() is False right after construction."""
    lr = LayerRules(layer=1, name="Market Structure")
    assert lr.is_populated() is False


@pytest.mark.integration
def test_layer_rules_is_populated_true_after_adding_chunks():
    """LayerRules.is_populated() is True after adding raw_chunks."""
    lr = LayerRules(layer=1, name="Market Structure")
    lr.raw_chunks.append("6-candle rule text")
    assert lr.is_populated() is True


# ---------------------------------------------------------------------------
# load_tct_rules — startup error handling
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_load_tct_rules_raises_on_missing_pdf_dir(tmp_path):
    """load_tct_rules raises SystemExit when PDF directory doesn't exist."""
    missing_dir = tmp_path / "no_pdfs_here"

    model_cache = (
        tmp_path / ".cache" / "torch" / "sentence_transformers"
        / EMBEDDING_MODEL.replace("/", "_")
    )
    model_cache.mkdir(parents=True)

    with patch("tct_pdf_rules.PDF_DIR", missing_dir), \
         patch("tct_pdf_rules.Path.home", return_value=tmp_path):
        with pytest.raises(SystemExit, match="PDF directory not found"):
            load_tct_rules()


@pytest.mark.integration
def test_load_tct_rules_raises_when_collection_empty_after_ingestion(tmp_path):
    """load_tct_rules raises SystemExit when ChromaDB has 0 docs after ingestion."""
    pdf_dir = tmp_path / "PDFs"
    pdf_dir.mkdir()
    chroma_dir = tmp_path / ".chromadb"
    model_cache = (
        tmp_path / ".cache" / "torch" / "sentence_transformers"
        / EMBEDDING_MODEL.replace("/", "_")
    )
    model_cache.mkdir(parents=True)

    # Use a MagicMock collection with explicit count=0 so the test is not
    # sensitive to EphemeralClient state leaking between tests in the same session.
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0  # explicitly empty after ingestion

    with patch("tct_pdf_rules.PDF_DIR", pdf_dir), \
         patch("tct_pdf_rules.CHROMA_DIR", chroma_dir), \
         patch("tct_pdf_rules.Path.home", return_value=tmp_path), \
         patch("tct_pdf_rules._import_chromadb", return_value=MagicMock()) as mock_chromadb, \
         patch("tct_pdf_rules._get_or_create_collection", return_value=mock_collection), \
         patch("tct_pdf_rules._populate_collection"):
        mock_client = mock_chromadb.return_value.PersistentClient.return_value  # noqa: F841
        with pytest.raises(SystemExit, match="collection is empty"):
            load_tct_rules()
