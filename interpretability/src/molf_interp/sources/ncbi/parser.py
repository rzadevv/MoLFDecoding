"""Pure-function PubMed XML parsers — no I/O, no network."""

from __future__ import annotations

from lxml import etree

from molf_interp.sources.ncbi.exceptions import MalformedResponseError
from molf_interp.sources.ncbi.models import ESearchResult, PubMedAbstract, PubMedAuthor
from molf_interp.utils.logging import get_logger

_PARSER = etree.XMLParser(resolve_entities=False, no_network=True)

log = get_logger(__name__)


def _parse_xml(xml_text: str) -> etree._Element:
    """Parse XML text with a safe parser.

    Args:
        xml_text: Raw XML string.

    Returns:
        Root lxml element.

    Raises:
        MalformedResponseError: On XML syntax error.
    """
    try:
        return etree.fromstring(xml_text.encode("utf-8"), parser=_PARSER)
    except etree.XMLSyntaxError as exc:
        raise MalformedResponseError(f"Invalid XML: {exc}") from exc


def parse_esearch_response(xml_text: str) -> ESearchResult:
    """Parse an ESearch XML response.

    Extracts <Count>, <IdList>/<Id> entries, and (if usehistory=y was used)
    <WebEnv> and <QueryKey>. Raises MalformedResponseError on unexpected structure.

    Args:
        xml_text: Raw ESearch XML response text.

    Returns:
        Parsed ESearchResult.

    Raises:
        MalformedResponseError: On unexpected XML structure or missing required elements.
    """
    root = _parse_xml(xml_text)

    # NCBI sometimes returns 200 with an <ERROR> body
    error_text = root.findtext("ERROR")
    if error_text:
        raise MalformedResponseError(f"NCBI returned error: {error_text}")

    count_text = root.findtext("Count")
    if count_text is None:
        raise MalformedResponseError("ESearch XML missing <Count>")
    try:
        count = int(count_text.strip())
    except ValueError as exc:
        raise MalformedResponseError(f"ESearch <Count> is not an integer: {count_text!r}") from exc

    pmids = tuple(el.text.strip() for el in root.findall("IdList/Id") if el.text)
    webenv = root.findtext("WebEnv")
    query_key = root.findtext("QueryKey")

    return ESearchResult(
        count=count,
        pmids=pmids,
        webenv=webenv or None,
        query_key=query_key or None,
    )


def parse_efetch_pubmed_response(xml_text: str) -> tuple[PubMedAbstract, ...]:
    """Parse an EFetch (db=pubmed) XML response containing multiple PubmedArticle entries.

    Tolerates missing abstract (returns empty string), missing year (None), structured
    abstracts (joins all AbstractText sections with labels), various author name formats.
    Skips records with no PMID and logs a warning.

    Args:
        xml_text: Raw EFetch XML response text.

    Returns:
        Tuple of PubMedAbstract instances (may be empty).

    Raises:
        MalformedResponseError: On XML syntax error.
    """
    root = _parse_xml(xml_text)

    # Handle both <PubmedArticleSet> and bare <PubmedArticle>
    articles = [root] if root.tag == "PubmedArticle" else root.findall(".//PubmedArticle")

    results: list[PubMedAbstract] = []
    for article in articles:
        medline = article.find("MedlineCitation")
        if medline is None:
            log.debug("Skipping PubmedArticle with no MedlineCitation")
            continue

        pmid_el = medline.find("PMID")
        if pmid_el is None or not pmid_el.text:
            log.warning("Skipping PubmedArticle with no <PMID>")
            continue
        pmid = pmid_el.text.strip()

        article_el = medline.find("Article")
        if article_el is None:
            log.debug("Skipping PMID {} — no <Article> element", pmid)
            continue

        # Title
        title_el = article_el.find("ArticleTitle")
        title = (title_el.text or "").strip() if title_el is not None else ""

        # Abstract — join all AbstractText sections with their labels
        abstract_parts: list[str] = []
        abstract_el = article_el.find("Abstract")
        if abstract_el is not None:
            for at in abstract_el.findall("AbstractText"):
                label = at.get("Label") or at.get("NlmCategory") or ""
                text = at.text or ""
                if label:
                    abstract_parts.append(f"{label}: {text.strip()}")
                else:
                    abstract_parts.append(text.strip())
        abstract = "\n\n".join(p for p in abstract_parts if p)

        # Skip records where both title and abstract are empty
        if not title and not abstract:
            log.debug("Skipping PMID {} — empty title and abstract", pmid)
            continue

        # Journal
        journal_el = article_el.find("Journal/Title")
        journal = (journal_el.text or "").strip() if journal_el is not None else None

        # Year — try multiple locations
        pub_year: int | None = None
        for xpath in (
            "Journal/JournalIssue/PubDate/Year",
            "Journal/JournalIssue/PubDate/MedlineDate",
            "ArticleDate/Year",
        ):
            year_el = article_el.find(xpath)
            if year_el is not None and year_el.text:
                raw = year_el.text.strip()[:4]
                try:
                    pub_year = int(raw)
                    break
                except ValueError:
                    pass

        # Authors
        authors: list[PubMedAuthor] = []
        for author_el in article_el.findall("AuthorList/Author"):
            last = author_el.findtext("LastName")
            fore = author_el.findtext("ForeName") or author_el.findtext("Initials")
            authors.append(PubMedAuthor(last_name=last, fore_name=fore))

        # MeSH terms
        mesh_terms = tuple(
            el.text.strip()
            for el in medline.findall("MeshHeadingList/MeshHeading/DescriptorName")
            if el.text
        )

        # Keywords
        keywords = tuple(
            el.text.strip() for el in medline.findall(".//KeywordList/Keyword") if el.text
        )

        # DOI
        doi: str | None = None
        for aid in article.findall(".//ArticleIdList/ArticleId"):
            if aid.get("IdType") == "doi" and aid.text:
                doi = aid.text.strip()
                break

        results.append(
            PubMedAbstract(
                pmid=pmid,
                title=title,
                abstract=abstract,
                journal=journal or None,
                pub_year=pub_year,
                authors=tuple(authors),
                mesh_terms=mesh_terms,
                keywords=keywords,
                doi=doi,
            )
        )

    return tuple(results)
