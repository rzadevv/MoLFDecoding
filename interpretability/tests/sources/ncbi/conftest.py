"""Shared fixtures for NCBI tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import respx

from molf_interp.sources.ncbi.config import NCBIConfig


@pytest.fixture()
def ncbi_config(tmp_path: Path) -> NCBIConfig:
    """NCBIConfig with fast settings and tmp_path cache."""
    return NCBIConfig(
        cache_dir=tmp_path / "pubmed_cache",
        rate_limit_per_second=1000,
        rate_limit_burst=1000,
        rate_limit_no_key=1000,
        max_retries=1,
        timeout_seconds=5.0,
    )


@pytest.fixture()
def respx_ncbi() -> respx.MockRouter:
    """Pre-configured respx router for NCBI E-utilities."""
    with respx.mock(
        base_url="https://eutils.ncbi.nlm.nih.gov",
        assert_all_called=False,
    ) as router:
        yield router


def fake_esearch_xml(
    count: int = 3,
    pmids: list[str] | None = None,
    webenv: str | None = "MCID_testwebenv",
    query_key: str | None = "1",
) -> str:
    """Build a realistic ESearch XML response.

    Args:
        count: Total record count.
        pmids: PMIDs to include in IdList.
        webenv: WebEnv value (None to omit).
        query_key: QueryKey value (None to omit).

    Returns:
        ESearch XML string.
    """
    if pmids is None:
        pmids = [f"3900000{i}" for i in range(1, count + 1)]

    id_list = "\n".join(f"    <Id>{pmid}</Id>" for pmid in pmids)
    webenv_el = f"  <WebEnv>{webenv}</WebEnv>" if webenv else ""
    qkey_el = f"  <QueryKey>{query_key}</QueryKey>" if query_key else ""

    return f"""\
<?xml version="1.0" encoding="UTF-8"?>
<eSearchResult>
  <Count>{count}</Count>
  <RetMax>{len(pmids)}</RetMax>
  <RetStart>0</RetStart>
  <IdList>
{id_list}
  </IdList>
{webenv_el}
{qkey_el}
</eSearchResult>"""


def fake_efetch_xml(abstracts: list[dict[str, object]] | None = None) -> str:
    """Build a realistic EFetch XML response.

    Each abstract dict may contain: pmid, title, abstract_text (str or list[dict]),
    journal, year, authors (list[dict with last_name/fore_name]), mesh_terms, doi.

    Args:
        abstracts: List of abstract dicts. Defaults to one sample record.

    Returns:
        EFetch PubmedArticleSet XML string.
    """
    if abstracts is None:
        abstracts = [
            {
                "pmid": "39000001",
                "title": "A study of necrosis and fibrosis",
                "abstract_text": "We studied necrosis patterns in carcinoma tissue.",
                "journal": "Test Journal",
                "year": "2024",
                "doi": "10.1000/test",
            }
        ]

    articles_xml = []
    for ab in abstracts:
        pmid = ab.get("pmid", "00000000")
        title = ab.get("title", "")
        journal = ab.get("journal", "")
        year = ab.get("year", "")
        doi = ab.get("doi", "")

        # Abstract — may be str or list of structured dicts
        abstract_raw = ab.get("abstract_text", "")
        if isinstance(abstract_raw, list):
            abstract_parts = "\n".join(
                f'          <AbstractText Label="{p["label"]}">{p["text"]}</AbstractText>'
                for p in abstract_raw  # type: ignore[union-attr]
            )
            abstract_xml = f"        <Abstract>\n{abstract_parts}\n        </Abstract>"
        else:
            abstract_xml = (
                f"        <Abstract>\n"
                f"          <AbstractText>{abstract_raw}</AbstractText>\n"
                f"        </Abstract>"
            )

        # Authors
        author_list = ""
        for auth in ab.get("authors", []):  # type: ignore[union-attr]
            author_list += (
                f"          <Author ValidYN='Y'>"
                f"<LastName>{auth.get('last_name', '')}</LastName>"  # type: ignore[union-attr]
                f"<ForeName>{auth.get('fore_name', '')}</ForeName>"  # type: ignore[union-attr]
                f"</Author>\n"
            )

        # MeSH terms
        mesh_xml = ""
        for term in ab.get("mesh_terms", []):  # type: ignore[union-attr]
            mesh_xml += (
                f"        <MeshHeading><DescriptorName>{term}</DescriptorName></MeshHeading>\n"
            )

        # Keywords
        kw_xml = ""
        for kw in ab.get("keywords", []):  # type: ignore[union-attr]
            kw_xml += f"      <Keyword>{kw}</Keyword>\n"

        articles_xml.append(
            f"""  <PubmedArticle>
    <MedlineCitation>
      <PMID Version="1">{pmid}</PMID>
      <Article>
        <Journal>
          <JournalIssue>
            <PubDate><Year>{year}</Year></PubDate>
          </JournalIssue>
          <Title>{journal}</Title>
        </Journal>
        <ArticleTitle>{title}</ArticleTitle>
{abstract_xml}
        <AuthorList>
{author_list}        </AuthorList>
      </Article>
      <MeshHeadingList>
{mesh_xml}      </MeshHeadingList>
      <KeywordList Owner="NOTNLM">
{kw_xml}      </KeywordList>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="doi">{doi}</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>"""
        )

    return (
        "<?xml version='1.0'?>\n<PubmedArticleSet>\n"
        + "\n".join(articles_xml)
        + "\n</PubmedArticleSet>"
    )
