"""
tct_pdf_rules.py — TCT Lecture Rule Extraction via ChromaDB RAG

Extracts and caches gate rules from the 6 core TCT lecture PDFs at startup.
Rules are extracted once, stored as dataclasses, and used deterministically
by phemex_tct_algo.py throughout the trading session.

Lecture mapping:
  Layer 1 → Market Structure  (Lecture 1)
  Layer 2 → Ranges            (Lecture 2)
  Layer 3 → Supply & Demand   (Lecture 3)
  Layer 4 → Liquidity         (Lecture 4)
  Layer 5 → TCT Schematics    (Lecture 5A + 5B)
  Layer 6 → Advanced TCT      (Lecture 6)

Design decisions:
  - Startup-only extraction: ChromaDB is never queried during the trading loop.
  - Explicit model: all-MiniLM-L6-v2 must be pre-cached locally before startup.
  - Single collection: "tct_lectures" with lecture_layer metadata for filtering.
  - Fail loudly on startup errors: missing model or PDF raises RuntimeError.
"""

from __future__ import annotations

import os
import sys
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pypdf import PdfReader

logger = logging.getLogger("TCT-PDFRules")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Path to PDF directory (resolved relative to this file)
PDF_DIR = Path(__file__).parent / "PDFs"

# ChromaDB persists to disk so embeddings survive restarts
CHROMA_DIR = Path(__file__).parent / ".chromadb"

COLLECTION_NAME = "tct_lectures"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunk size in characters (~500 tokens at ~4 chars/token)
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200

# Each lecture PDF maps to one gate layer (1–6).
# Lecture 5 spans two files (5A and 5B), both mapped to layer 5.
LECTURE_PDF_MAP: dict[int, list[str]] = {
    1: ["TCT mentorship - Lecture 1 Market Structure8Pages.pdf",
        "TCT-mentorship-Lecture-1-Market-Structure.pdf"],
    2: ["TCT-2024-mentorship-Lecture-2-Ranges.pdf"],
    3: ["TCT-2024-mentorship-Lecture-3-Supply-AND-Demand.pdf",
        "TCT-mentorship-Lecture-3-Supply-and-Demand.pdf"],
    4: ["TCT-2024-mentorship-Lecture-4-Liquidity.pdf"],
    5: ["TCT-2024-mentorship-Lecture-5A-TCT-schematics.pdf",
        "TCT-2024-mentorship-Lecture-5B-TCT-schematics.pdf"],
    6: ["TCT-2024-mentorship-Lecture-6-Advanced-TCT-schematics.pdf"],
}

# Queries sent to ChromaDB to pull the key rules for each layer
LAYER_QUERIES: dict[int, list[str]] = {
    1: [
        "6-candle rule pivot detection inside bar exclusion",
        "MSH MSL confirmation revisit rule",
        "BOS break of structure good bad quality",
        "CHoCH change of character domino effect",
        "level 1 2 3 structure hierarchy trend classification",
    ],
    2: [
        "range high range low definition valid range",
        "range duration minimum candles",
        "range boundary tap sequence",
        "range displacement internal vs external",
    ],
    3: [
        "supply zone demand zone definition entry",
        "order block OB confirmation flip",
        "fair value gap FVG mitigation",
        "breaker block supply demand reversal",
    ],
    4: [
        "liquidity sweep engineering liquidity",
        "buy side sell side liquidity equal highs lows",
        "liquidity curve accumulation distribution",
        "stop hunt run on liquidity confirmation",
    ],
    5: [
        "Model 1 schematic entry three taps BOS",
        "Model 2 schematic structure",
        "TCT three condition trigger qualification",
        "entry trigger confirmation sequence",
        "HTF MTF LTF alignment requirement",
    ],
    6: [
        "advanced TCT pattern extension",
        "nested model confirmation",
        "high probability bias HPB definition",
        "confluence requirements full entry",
    ],
}


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class LayerRules:
    """Rules extracted from ChromaDB for one lecture layer."""

    layer: int
    name: str
    raw_chunks: list[str] = field(default_factory=list)
    # Populated after extraction: condensed rule text per query topic
    rules_by_topic: dict[str, str] = field(default_factory=dict)

    def is_populated(self) -> bool:
        return bool(self.raw_chunks)


@dataclass
class TCTRuleSet:
    """Complete rule set for all 6 layers, extracted at startup."""

    layers: dict[int, LayerRules] = field(default_factory=dict)

    def get(self, layer: int) -> Optional[LayerRules]:
        return self.layers.get(layer)

    def all_populated(self) -> bool:
        return all(lr.is_populated() for lr in self.layers.values())


LAYER_NAMES = {
    1: "Market Structure",
    2: "Ranges",
    3: "Supply & Demand",
    4: "Liquidity",
    5: "TCT Schematics",
    6: "Advanced TCT",
}


# ---------------------------------------------------------------------------
# Pre-flight check
# ---------------------------------------------------------------------------

def _check_embedding_model_cached() -> None:
    """
    Verify that the sentence-transformers model is already downloaded locally.

    If the model cache directory doesn't exist, fail loudly with an actionable
    error message rather than attempting a download that might hang or fail
    mid-startup on a restricted network.
    """
    # sentence-transformers caches to ~/.cache/torch/sentence_transformers
    # or to SENTENCE_TRANSFORMERS_HOME if set.
    cache_base = Path(
        os.environ.get("SENTENCE_TRANSFORMERS_HOME",
                       Path.home() / ".cache" / "torch" / "sentence_transformers")
    )
    model_cache = cache_base / EMBEDDING_MODEL.replace("/", "_")

    if not model_cache.exists():
        raise RuntimeError(
            f"\n[FATAL] Embedding model not cached locally.\n"
            f"Expected: {model_cache}\n\n"
            f"Pre-download the model before starting the server:\n"
            f"  python -c \"from sentence_transformers import SentenceTransformer; "
            f"SentenceTransformer('{EMBEDDING_MODEL}')\"\n"
        )

    logger.info("Embedding model cache verified: %s", model_cache)


# ---------------------------------------------------------------------------
# PDF loading helpers
# ---------------------------------------------------------------------------

def _extract_pdf_text(pdf_path: Path) -> str:
    """Extract full text from a PDF file using pypdf."""
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())
    return "\n\n".join(pages)


def _chunk_text(text: str) -> list[str]:
    """
    Split text into overlapping chunks of CHUNK_SIZE characters.

    Overlap prevents rule context from being cut at chunk boundaries.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# ---------------------------------------------------------------------------
# ChromaDB population
# ---------------------------------------------------------------------------

def _import_chromadb():
    """Lazy-import chromadb so onnxruntime (~200-300 MB) is not loaded at module import time."""
    import chromadb  # noqa: PLC0415
    return chromadb

def _get_or_create_collection(client: "chromadb.Client") -> "chromadb.Collection":
    """Return the tct_lectures collection, creating it if it doesn't exist."""
    # Lazy imports: chromadb + onnxruntime are heavy (~200-300 MB).
    # Importing here instead of at module level prevents the memory spike from
    # occurring whenever phemex_tct_trader/phemex_tct_algo are imported by route handlers.
    chromadb = _import_chromadb()
    from chromadb.utils import embedding_functions  # noqa: F811
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


def _populate_collection(
    collection: chromadb.Collection,
    force_repopulate: bool = False,
) -> None:
    """
    Load all lecture PDFs into the ChromaDB collection.

    Skips re-ingestion if the collection already has documents, unless
    force_repopulate=True.
    """
    if collection.count() > 0 and not force_repopulate:
        logger.info(
            "Collection '%s' already has %d documents — skipping ingestion.",
            COLLECTION_NAME,
            collection.count(),
        )
        return

    logger.info("Populating ChromaDB collection from PDFs in %s", PDF_DIR)

    for layer, filenames in LECTURE_PDF_MAP.items():
        for filename in filenames:
            pdf_path = PDF_DIR / filename
            if not pdf_path.exists():
                logger.warning("PDF not found, skipping: %s", pdf_path)
                continue

            logger.info("Ingesting Layer %d: %s", layer, filename)
            text = _extract_pdf_text(pdf_path)
            if not text.strip():
                logger.warning("No text extracted from %s", filename)
                continue

            chunks = _chunk_text(text)
            ids = [f"L{layer}_{filename[:20]}_{i}" for i, _ in enumerate(chunks)]
            metadatas = [
                {"lecture_layer": layer, "source_file": filename, "chunk_index": i}
                for i, _ in enumerate(chunks)
            ]

            collection.upsert(
                ids=ids,
                documents=chunks,
                metadatas=metadatas,
            )
            logger.info("  → %d chunks ingested for Layer %d", len(chunks), layer)

    logger.info("Collection populated. Total documents: %d", collection.count())


# ---------------------------------------------------------------------------
# Rule extraction
# ---------------------------------------------------------------------------

def _query_layer_rules(
    collection: chromadb.Collection,
    layer: int,
    n_results: int = 3,
) -> LayerRules:
    """
    Query ChromaDB for each topic in the given layer and assemble a LayerRules.

    n_results: number of chunks to retrieve per query topic. More chunks means
    richer context but slower startup. 3 is sufficient for deterministic rules.
    """
    lr = LayerRules(layer=layer, name=LAYER_NAMES[layer])
    queries = LAYER_QUERIES[layer]

    for query in queries:
        results = collection.query(
            query_texts=[query],
            n_results=n_results,
            where={"lecture_layer": layer},
        )
        docs = results.get("documents", [[]])[0]
        combined = "\n---\n".join(docs)
        lr.rules_by_topic[query] = combined
        lr.raw_chunks.extend(docs)

    logger.info(
        "Layer %d (%s): %d rule chunks extracted across %d topics.",
        layer, LAYER_NAMES[layer], len(lr.raw_chunks), len(queries),
    )
    return lr


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_tct_rules(force_repopulate: bool = False) -> TCTRuleSet:
    """
    Entry point: verify model cache, populate ChromaDB, extract all rules.

    Called once at server startup. Returns a TCTRuleSet that the trading
    loop uses without ever touching ChromaDB again.

    Raises RuntimeError on any unrecoverable startup error (missing model,
    empty PDF directory, zero documents extracted).
    """
    _check_embedding_model_cached()

    if not PDF_DIR.exists():
        raise RuntimeError(
            f"[FATAL] PDF directory not found: {PDF_DIR}\n"
            "Place the TCT lecture PDFs in the PDFs/ directory."
        )

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    chromadb = _import_chromadb()
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = _get_or_create_collection(client)

    _populate_collection(collection, force_repopulate=force_repopulate)

    if collection.count() == 0:
        raise RuntimeError(
            "[FATAL] ChromaDB collection is empty after ingestion. "
            "Check that PDFs contain extractable text."
        )

    rule_set = TCTRuleSet()
    for layer in LECTURE_PDF_MAP:
        rule_set.layers[layer] = _query_layer_rules(collection, layer)

    if not rule_set.all_populated():
        unpopulated = [
            layer for layer, lr in rule_set.layers.items()
            if not lr.is_populated()
        ]
        raise RuntimeError(
            f"[FATAL] Rule extraction incomplete. Empty layers: {unpopulated}\n"
            "Verify PDFs for those layers contain extractable text."
        )

    logger.info("TCT rule extraction complete. All 6 layers populated.")
    return rule_set
