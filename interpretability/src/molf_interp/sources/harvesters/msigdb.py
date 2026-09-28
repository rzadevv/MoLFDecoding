"""MSigDB GMT harvester — sync HTTPS download + line-by-line parse."""

from __future__ import annotations

import time
from enum import StrEnum
from pathlib import Path

import httpx
from loguru import logger
from pydantic import Field

from molf_interp.io.config import BaseConfig
from molf_interp.sources.raw_concept import CandidateTier, RawConcept, write_raw_concepts

_KNOWN_PREFIXES: frozenset[str] = frozenset(
    {"HALLMARK", "REACTOME", "KEGG", "GOBP", "GOCC", "GOMF"}
)


class MSigDBCollection(StrEnum):
    """MSigDB collections to harvest."""

    HALLMARK = "h.all"
    C2_REACTOME = "c2.cp.reactome"
    C2_KEGG_LEGACY = "c2.cp.kegg_legacy"


class MSigDBHarvestConfig(BaseConfig):
    """Configuration for MSigDB GMT download and parse."""

    msigdb_version: str = "2024.1.Hs"
    collections: tuple[MSigDBCollection, ...] = (
        MSigDBCollection.HALLMARK,
        MSigDBCollection.C2_REACTOME,
        MSigDBCollection.C2_KEGG_LEGACY,
    )
    base_url: str = "https://data.broadinstitute.org/gsea-msigdb/msigdb/release"
    cache_dir: Path = Path("data/cache/msigdb")
    output_path: Path = Path("data/cache/harvest/msigdb.parquet")
    timeout_seconds: float = 60.0
    candidate_tier_by_collection: dict[str, CandidateTier] = Field(
        default_factory=lambda: {
            "h.all": CandidateTier.H2_GENE_PROGRAM,
            "c2.cp.reactome": CandidateTier.H2_PATHWAY,
            "c2.cp.kegg_legacy": CandidateTier.H2_PATHWAY,
        }
    )


def humanize_msigdb_set_name(name: str) -> str:
    """Convert 'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION' → 'Epithelial mesenchymal transition'.

    Strips the collection prefix (e.g. HALLMARK_, REACTOME_, KEGG_), replaces
    underscores with spaces, and applies sentence case. Note: acronyms like 'TCA'
    are lowercased (imperfect but acceptable — curation can refine).

    Args:
        name: Raw MSigDB set name in COLLECTION_SCREAMING_SNAKE_CASE format.

    Returns:
        Human-readable label in sentence case.
    """
    _, _, body = name.partition("_")
    if not body:
        body = name
    human = body.replace("_", " ")
    if not human:
        return ""
    return human[0].upper() + human[1:].lower()


def _download_gmt(url: str, dest: Path, timeout: float) -> None:
    """Download a GMT file to dest via HTTP GET.

    Args:
        url: Full URL of the GMT file.
        dest: Destination file path.
        timeout: Request timeout in seconds.

    Raises:
        FileNotFoundError: If the server returns 404.
        httpx.HTTPStatusError: On other non-2xx responses.
    """
    with httpx.Client(timeout=timeout) as http:
        resp = http.get(url)
        if resp.status_code == 404:
            raise FileNotFoundError(f"MSigDB GMT not found at {url!r}. Check msigdb_version.")
        resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(resp.text, encoding="utf-8")


def _parse_gmt(
    path: Path, collection: MSigDBCollection, version: str, tier: CandidateTier
) -> list[RawConcept]:
    """Parse a GMT file into RawConcept records.

    Args:
        path: Path to the cached GMT file.
        collection: The MSigDB collection enum value.
        version: MSigDB version string (stored in extra).
        tier: CandidateTier to assign to all records.

    Returns:
        List of RawConcept instances (sets with no genes are skipped).
    """
    source_name = f"msigdb_{collection.value}"
    records: list[RawConcept] = []
    with path.open(encoding="utf-8") as fh:
        for line_num, raw_line in enumerate(fh, start=1):
            line = raw_line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                logger.warning(
                    "Skipping malformed GMT line {} in {}: fewer than 2 fields",
                    line_num,
                    path.name,
                )
                continue
            set_name = parts[0]
            set_url = parts[1]
            genes = [g.strip() for g in parts[2:] if g.strip()]
            if not genes:
                logger.warning(
                    "Skipping GMT set '{}': no gene symbols (line {})", set_name, line_num
                )
                continue
            records.append(
                RawConcept(
                    source_name=source_name,
                    source_id=set_name,
                    preferred_label=humanize_msigdb_set_name(set_name),
                    synonyms=(set_name,),
                    definition=None,
                    parent_ids=(),
                    candidate_tier=tier,
                    extra={
                        "genes": ",".join(genes),
                        "msigdb_url": set_url,
                        "msigdb_version": version,
                    },
                )
            )
    return records


def harvest_msigdb(config: MSigDBHarvestConfig) -> int:
    """Download and parse MSigDB GMT collections into RawConcept records.

    Downloads each collection GMT file to config.cache_dir if not already cached
    (checked by file existence). Parses line-by-line; skips gene sets with no
    gene symbols. Writes all collections to config.output_path.

    Args:
        config: Harvest configuration.

    Returns:
        Total number of gene-set RawConcept records written.
    """
    started = time.perf_counter()
    all_records: list[RawConcept] = []

    for collection in config.collections:
        filename = f"{collection.value}.v{config.msigdb_version}.symbols.gmt"
        cache_file = config.cache_dir / filename
        url = f"{config.base_url}/{config.msigdb_version}/{filename}"

        if not cache_file.exists():
            logger.info("Downloading {} → {}", url, cache_file)
            _download_gmt(url, cache_file, config.timeout_seconds)
        else:
            logger.info("Using cached GMT: {}", cache_file)

        tier = config.candidate_tier_by_collection.get(
            collection.value, CandidateTier.H2_GENE_PROGRAM
        )
        records = _parse_gmt(cache_file, collection, config.msigdb_version, tier)
        logger.info("MSigDB {}: {} gene sets parsed", collection.value, len(records))
        all_records.extend(records)

    write_raw_concepts(all_records, config.output_path)
    elapsed = time.perf_counter() - started
    logger.info(
        "MSigDB harvest complete: {} total records in {:.1f}s",
        len(all_records),
        elapsed,
    )
    return len(all_records)
