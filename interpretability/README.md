# molf-interp: concept bank and CONCH embedding

Python package that builds the pathology concept vocabulary used to interpret
the MoLF experts and embeds it with the CONCH text encoder. How the vocabulary
was assembled, audited and cleaned is described in
[`../docs/concept_bank.md`](../docs/concept_bank.md); scoring the experts
against it is in [`../experiments/interpretability/`](../experiments/interpretability/).

## Pipeline

1. **Harvesting.** Candidate concepts from BioPortal (NCIt, SNOMED CT, Cell
   Ontology, GO), MSigDB gene sets, and PubMed abstracts (NCBI E-utilities
   with scispaCy NER); UBERON, also from BioPortal, supplies organ names for
   curation. Responses are cached; the first run takes about 30 minutes.
2. **Curation**, in seven stages: SapBERT embedding of the labels,
   cross-source deduplication (HDBSCAN), a filter that compares each label
   with the reference vocabulary of its tier and with non-visual negative
   classes, adjudication of borderline concepts by a local LLM
   (qwen2.5:3b-instruct via Ollama), tier assignment, label canonicalisation,
   and generation of human and mouse text prompts.
3. **Comparison** of the harvested bank with the pathologist-curated reference
   bank. The result is a merged bank and its primary subset, *View A*: the
   reference concepts plus the harvested concepts that match or closely
   resemble them. This step also writes a report and a UMAP of the two banks.
4. **Embedding.** View A is cleaned (clinical labels, duplicates, human-only
   mouse prompts) and its prompts are embedded with CONCH v1. The model layer
   is a small registry, but CONCH is the only model implemented.

## Setup

```bash
uv sync --dev
uv run pre-commit install
```

Harvesting and CONCH need credentials in a `.env` file in this directory
(gitignored):

```ini
BIOPORTAL_API_KEY=...
NCBI_API_KEY=...
NCBI_EMAIL=...
HF_TOKEN=...        # CONCH weights are gated on HuggingFace
```

## Usage

```bash
# Reference bank
uv run molf-interp concepts validate-reference
uv run molf-interp concepts build-reference

# Harvest and curate (--skip-llm skips the LLM adjudication stage)
uv run molf-interp harvest all
uv run molf-interp curate calibrate
uv run molf-interp curate run

# Compare with the reference bank and build View A
uv run molf-interp compare run
uv run molf-interp compare build-lexicon-view

# Clean View A and embed its prompts with CONCH
uv run molf-interp vlm prepare-view-a
uv run molf-interp vlm concept-embed
```

`data/outputs/` holds the concept bank and embeddings used for the reported
results. `compare run` rewrites `data/outputs/comparison/` in place, and
`vlm concept-embed` refuses to replace the existing embeddings unless given
`--force`. To keep the shipped files intact, change `output_dir` in
`configs/comparison/default.yaml` and `configs/vlm/concept_embedding.yaml`
before re-running.

`vlm prepare-view-a` also writes the cleaned prompt table
(`data/cache/concept_embeddings/view_a_ready.parquet`) that the expert analysis
reads alongside the embeddings. `scripts/analyze_view_a.py` prints summary
statistics of View A that were used for the concept-bank audit.

## Layout

| Path | Contents |
|---|---|
| `configs/` | YAML configuration for harvesting, curation, comparison and embedding |
| `data/concept_bank/` | the reference bank (TSV) and the script that generates its cross-species prompt table |
| `data/outputs/comparison/` | merged bank, View A membership, comparison report |
| `data/outputs/vlm/conch_text_embeddings/` | CONCH embeddings of all 5,803 View A prompts; the Run-2 analysis uses the 2,450 visual concepts with human prompts |
| `data/cache/` | regenerable caches (gitignored) |
| `src/molf_interp/` | the package: `sources/`, `curation/`, `comparison/`, `vlm/`, `concepts/`, `io/`, `cli/`, `utils/` |
| `tests/` | pytest suite |

## Development

```bash
uv run ruff format . && uv run ruff check .
uv run mypy src
uv run pytest -m "not integration"   # without tests that need real data or network access
```
