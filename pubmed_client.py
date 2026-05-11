from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import re
import time
from typing import Iterable, Optional
import xml.etree.ElementTree as ET

import pandas as pd
import requests


EUTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
TOOL_NAME = "ResearchMindAI"
DEFAULT_ESEARCH_FALLBACK_QUERIES = [
    "alzheimer artificial intelligence",
    "alzheimer disease artificial intelligence",
    "alzheimer disease deep learning",
    "alzheimer disease MRI",
    "artificial intelligence diagnosis",
]

RESEARCHMIND_COLUMNS = [
    "pmid", "doi", "title", "abstract", "journal", "pub_year", "pub_month",
    "pub_month_num", "month_year", "authors", "authors_count", "country",
    "research_type", "keywords", "major_topic", "language", "open_access",
    "source", "year",
]

MONTH_MAP = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


class PubMedClientError(Exception):
    """Base exception carrying a user-facing PubMed error message."""

    def __init__(self, message: str, user_message: Optional[str] = None):
        super().__init__(message)
        self.user_message = user_message or message


class PubMedConnectionError(PubMedClientError):
    pass


class PubMedHTTPError(PubMedClientError):
    pass


class PubMedRateLimitError(PubMedClientError):
    pass


class PubMedXMLParseError(PubMedClientError):
    pass


@dataclass(frozen=True)
class PubMedConfig:
    api_key: str = ""
    email: str = ""
    tool: str = TOOL_NAME
    timeout: int = 30
    max_retries: int = 3
    retry_backoff_seconds: float = 1.0

    @property
    def requests_per_second(self) -> float:
        return 10.0 if self.api_key else 3.0


def _read_env_file(path: str | Path = ".env") -> dict[str, str]:
    env_path = Path(path)

    if not env_path.exists():
        return {}

    values: dict[str, str] = {}

    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values

    for line in lines:
        stripped = line.strip()

        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key:
            values[key] = value

    return values


def get_pubmed_config(env_path: str | Path = ".env") -> PubMedConfig:
    env_file = _read_env_file(env_path)

    return PubMedConfig(
        api_key=os.getenv("PUBMED_API_KEY", env_file.get("PUBMED_API_KEY", "")).strip(),
        email=os.getenv("NCBI_EMAIL", env_file.get("NCBI_EMAIL", "")).strip(),
    )


def get_pubmed_config_warning(env_path: str | Path = ".env") -> str:
    config = get_pubmed_config(env_path)

    if config.email:
        return ""

    return (
        "NCBI_EMAIL is not set. PubMed Live will still run, but NCBI "
        "recommends sending an email parameter so they can contact the "
        "application owner if traffic causes problems."
    )


class PubMedClient:
    def __init__(
        self,
        config: Optional[PubMedConfig] = None,
        session: Optional[requests.Session] = None,
    ):
        self.config = config or get_pubmed_config()
        self.session = session or requests.Session()
        self._last_request_at = 0.0
        self.last_search_metadata: dict[str, object] = {}

    def search(
        self,
        query: str,
        max_results: int = 10,
        years_back: Optional[int] = None,
    ) -> list[dict]:
        pmids = self.esearch(
            query=query,
            max_results=max_results,
            years_back=years_back,
        )

        if not pmids:
            return []

        return self.efetch(pmids)

    def esearch(
        self,
        query: str,
        max_results: int = 10,
        years_back: Optional[int] = None,
    ) -> list[str]:
        if not str(query).strip():
            return []

        candidates = _dedupe_queries([query, *DEFAULT_ESEARCH_FALLBACK_QUERIES])
        self.last_search_metadata = {
            "original_query": str(query).strip(),
            "tried_queries": [],
            "successful_query": "",
            "final_status": "pending",
            "error": "",
            "service_unavailable": False,
        }
        last_backend_error = ""

        for candidate in candidates:
            self.last_search_metadata["tried_queries"].append(candidate)

            try:
                pmids = self._esearch_once(
                    query=candidate,
                    max_results=max_results,
                    years_back=years_back,
                )
            except PubMedHTTPError as exc:
                if _is_esearch_backend_error(str(exc)) or _is_esearch_backend_error(exc.user_message):
                    last_backend_error = exc.user_message
                    self.last_search_metadata["error"] = exc.user_message
                    continue
                raise

            self.last_search_metadata["successful_query"] = candidate
            self.last_search_metadata["final_status"] = "success"
            return pmids

        if last_backend_error:
            self.last_search_metadata["final_status"] = "service_unavailable"
            self.last_search_metadata["service_unavailable"] = True
            self.last_search_metadata["error"] = last_backend_error
            return []

        self.last_search_metadata["final_status"] = "no_results"
        return []

    def _esearch_once(
        self,
        query: str,
        max_results: int = 10,
        years_back: Optional[int] = None,
    ) -> list[str]:
        if not str(query).strip():
            return []

        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max(0, int(max_results)),
            "retmode": "json",
            "sort": "relevance",
        }

        if years_back is not None:
            current_year = datetime.now().year
            safe_years_back = max(1, min(int(years_back), 5))
            params.update({
                "datetype": "pdat",
                "mindate": str(current_year - safe_years_back + 1),
                "maxdate": str(current_year),
            })

        params.update(self._identity_params())

        response = self._request("esearch.fcgi", params=params)

        try:
            payload = response.json()
        except ValueError as exc:
            raise PubMedXMLParseError(
                "NCBI ESearch response could not be parsed.",
                "PubMed search response could not be parsed. Please try again later.",
            ) from exc

        error = payload.get("error") or payload.get("esearchresult", {}).get("ERROR")
        if error:
            raise PubMedHTTPError(
                f"NCBI ESearch returned an error: {error}",
                f"PubMed search failed: {error}",
            )

        return [str(pmid) for pmid in payload.get("esearchresult", {}).get("idlist", [])]

    def efetch(self, pmids: Iterable[str]) -> list[dict]:
        pmid_list = [str(pmid).strip() for pmid in pmids if str(pmid).strip()]

        if not pmid_list:
            return []

        params = {
            "db": "pubmed",
            "id": ",".join(pmid_list),
            "retmode": "xml",
        }
        params.update(self._identity_params())

        response = self._request("efetch.fcgi", params=params)

        try:
            root = ET.fromstring(response.content)
        except ET.ParseError as exc:
            raise PubMedXMLParseError(
                "NCBI EFetch XML response could not be parsed.",
                "PubMed returned malformed XML. Please retry or reduce max results.",
            ) from exc

        error_nodes = root.findall(".//ERROR")
        if error_nodes:
            error_text = "; ".join(_node_text(node) for node in error_nodes if _node_text(node))
            raise PubMedHTTPError(
                f"NCBI EFetch returned an error: {error_text}",
                f"PubMed fetch failed: {error_text}",
            )

        return [_parse_pubmed_article(article) for article in root.findall(".//PubmedArticle")]

    def _identity_params(self) -> dict[str, str]:
        params = {"tool": self.config.tool}

        if self.config.email:
            params["email"] = self.config.email

        if self.config.api_key:
            params["api_key"] = self.config.api_key

        return params

    def _request(self, endpoint: str, params: dict) -> requests.Response:
        url = f"{EUTILS_BASE_URL}/{endpoint}"
        retryable_statuses = {429, 500, 502, 503, 504}

        for attempt in range(self.config.max_retries + 1):
            self._wait_for_rate_limit()

            try:
                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.config.timeout,
                )
            except requests.Timeout as exc:
                if attempt < self.config.max_retries:
                    self._sleep_before_retry(attempt)
                    continue

                raise PubMedConnectionError(
                    "NCBI request timed out.",
                    "PubMed connection timed out. Please retry with fewer results.",
                ) from exc
            except requests.exceptions.SSLError as exc:
                raise PubMedConnectionError(
                    f"NCBI SSL verification failed: {exc}",
                    (
                        "PubMed SSL certificate verification failed on this machine. "
                        "Check the local Python/requests certificate store and retry."
                    ),
                ) from exc
            except requests.RequestException as exc:
                if attempt < self.config.max_retries:
                    self._sleep_before_retry(attempt)
                    continue

                raise PubMedConnectionError(
                    f"NCBI connection failed: {exc}",
                    "PubMed connection failed. Please check your network and retry.",
                ) from exc

            if self._is_rate_limited(response):
                if attempt < self.config.max_retries:
                    self._sleep_before_retry(attempt)
                    continue

                raise PubMedRateLimitError(
                    "NCBI rate limit exceeded.",
                    (
                        "PubMed rate limit was reached. Wait a moment, reduce max "
                        "results, or set PUBMED_API_KEY in .env."
                    ),
                )

            if response.status_code in retryable_statuses and attempt < self.config.max_retries:
                self._sleep_before_retry(attempt)
                continue

            if response.status_code >= 400:
                raise PubMedHTTPError(
                    f"NCBI returned HTTP {response.status_code}: {response.text[:300]}",
                    f"PubMed returned HTTP {response.status_code}. Please retry later.",
                )

            return response

        raise PubMedConnectionError(
            "NCBI request failed after retries.",
            "PubMed request failed after retries. Please try again later.",
        )

    def _wait_for_rate_limit(self) -> None:
        min_interval = 1.0 / self.config.requests_per_second
        elapsed = time.monotonic() - self._last_request_at

        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        self._last_request_at = time.monotonic()

    def _sleep_before_retry(self, attempt: int) -> None:
        time.sleep(self.config.retry_backoff_seconds * (2 ** attempt))

    @staticmethod
    def _is_rate_limited(response: requests.Response) -> bool:
        text = response.text.lower()
        return response.status_code == 429 or "api rate limit exceeded" in text


def search_pubmed(
    query: str,
    max_results: int = 10,
    years_back: Optional[int] = None,
) -> pd.DataFrame:
    client = PubMedClient()

    try:
        records = client.search(
            query=query,
            max_results=max_results,
            years_back=years_back,
        )
        df = normalize_pubmed_to_researchmind_schema(pd.DataFrame(records))
        df.attrs["pubmed_metadata"] = client.last_search_metadata
        return df
    except PubMedClientError as exc:
        df = normalize_pubmed_to_researchmind_schema(pd.DataFrame())
        df.attrs["pubmed_metadata"] = {
            "original_query": str(query).strip(),
            "tried_queries": DEFAULT_ESEARCH_FALLBACK_QUERIES.copy(),
            "successful_query": "",
            "final_status": "service_unavailable",
            "service_unavailable": True,
            "error": exc.user_message,
        }
        return df


def normalize_pubmed_to_researchmind_schema(pubmed_df: pd.DataFrame) -> pd.DataFrame:
    df = pubmed_df.copy()

    for col in ["pmid", "doi", "title", "abstract", "year", "journal", "authors", "keywords"]:
        if col not in df.columns:
            df[col] = ""

    if "source" not in df.columns:
        df["source"] = "PubMed"
    else:
        df["source"] = df["source"].replace("", "PubMed").fillna("PubMed")

    if "publication_types" not in df.columns:
        df["publication_types"] = ""

    if "language" not in df.columns:
        df["language"] = "Unknown"

    if "pub_month" not in df.columns:
        df["pub_month"] = "Unknown"

    if "pub_month_num" not in df.columns:
        df["pub_month_num"] = 0

    df["pmid"] = df["pmid"].fillna("").astype(str)
    df["doi"] = df["doi"].fillna("").astype(str)
    df["title"] = df["title"].fillna("Unknown").replace("", "Unknown").astype(str)
    df["abstract"] = df["abstract"].fillna("Unknown").replace("", "Unknown").astype(str)
    df["journal"] = df["journal"].fillna("Unknown").replace("", "Unknown").astype(str)
    df["authors"] = df["authors"].fillna("").astype(str)
    df["keywords"] = df["keywords"].fillna("").astype(str)
    df["language"] = df["language"].fillna("Unknown").replace("", "Unknown").astype(str)

    year = pd.to_numeric(df["year"], errors="coerce")
    df["year"] = year.astype("Int64")
    df["pub_year"] = df["year"]

    df["pub_month"] = df["pub_month"].fillna("Unknown").replace("", "Unknown").astype(str)
    df["pub_month_num"] = pd.to_numeric(df["pub_month_num"], errors="coerce").fillna(0).astype(int)

    df["month_year"] = "Unknown-" + df["pub_year"].astype(str)
    df.loc[df["pub_year"].isna(), "month_year"] = "Unknown"

    df["authors_count"] = df["authors"].map(_count_authors)
    df["country"] = "Unknown"
    df["research_type"] = (
        df["publication_types"].fillna("").replace("", "Journal Article").astype(str)
    )
    df["major_topic"] = df["keywords"].map(_first_keyword).replace("", "Unknown")
    df["open_access"] = "Unknown"

    df = _deduplicate_pubmed(df)

    for col in RESEARCHMIND_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    return df[RESEARCHMIND_COLUMNS]


def _parse_pubmed_article(article: ET.Element) -> dict:
    year, month_text, month_num = _extract_pub_date(article)
    keywords = [_node_text(node) for node in article.findall(".//KeywordList/Keyword")]
    publication_types = [
        _node_text(node)
        for node in article.findall(".//PublicationTypeList/PublicationType")
    ]

    return {
        "pmid": _node_text(article.find("./MedlineCitation/PMID")),
        "doi": _extract_doi(article),
        "title": _node_text(article.find("./MedlineCitation/Article/ArticleTitle")),
        "abstract": _extract_abstract(article),
        "year": year,
        "journal": _extract_journal(article),
        "authors": "; ".join(_extract_authors(article)),
        "keywords": "; ".join(k for k in keywords if k),
        "source": "PubMed",
        "language": _node_text(article.find("./MedlineCitation/Article/Language")) or "Unknown",
        "publication_types": "; ".join(t for t in publication_types if t),
        "pub_month": month_text or "Unknown",
        "pub_month_num": month_num,
    }


def _extract_pub_date(article: ET.Element) -> tuple[Optional[int], str, int]:
    pub_date = article.find("./MedlineCitation/Article/Journal/JournalIssue/PubDate")

    if pub_date is None:
        pub_date = article.find("./MedlineCitation/Article/ArticleDate")

    year_text = _node_text(pub_date.find("Year")) if pub_date is not None else ""
    month_text = _node_text(pub_date.find("Month")) if pub_date is not None else ""

    if not year_text and pub_date is not None:
        medline_date = _node_text(pub_date.find("MedlineDate"))
        match = re.search(r"\b(19|20)\d{2}\b", medline_date)
        year_text = match.group(0) if match else ""

    try:
        year = int(year_text)
    except (TypeError, ValueError):
        year = None

    return year, month_text or "Unknown", _month_to_number(month_text)


def _extract_journal(article: ET.Element) -> str:
    title = _node_text(article.find("./MedlineCitation/Article/Journal/Title"))

    if title:
        return title

    return _node_text(article.find("./MedlineCitation/Article/Journal/ISOAbbreviation"))


def _extract_abstract(article: ET.Element) -> str:
    parts = []

    for node in article.findall("./MedlineCitation/Article/Abstract/AbstractText"):
        text = _node_text(node)

        if not text:
            continue

        label = node.attrib.get("Label", "").strip()
        parts.append(f"{label}: {text}" if label else text)

    return " ".join(parts)


def _extract_authors(article: ET.Element) -> list[str]:
    authors = []

    for author in article.findall("./MedlineCitation/Article/AuthorList/Author"):
        collective = _node_text(author.find("CollectiveName"))

        if collective:
            authors.append(collective)
            continue

        fore_name = _node_text(author.find("ForeName"))
        last_name = _node_text(author.find("LastName"))
        initials = _node_text(author.find("Initials"))

        name_parts = [part for part in [fore_name or initials, last_name] if part]

        if name_parts:
            authors.append(" ".join(name_parts))

    return authors


def _extract_doi(article: ET.Element) -> str:
    for node in article.findall(".//ArticleIdList/ArticleId"):
        if node.attrib.get("IdType", "").lower() == "doi":
            return _node_text(node)

    for node in article.findall(".//ELocationID"):
        if node.attrib.get("EIdType", "").lower() == "doi":
            return _node_text(node)

    return ""


def _node_text(node: Optional[ET.Element]) -> str:
    if node is None:
        return ""

    return re.sub(r"\s+", " ", "".join(node.itertext())).strip()


def _dedupe_queries(queries: Iterable[str]) -> list[str]:
    seen = set()
    clean_queries = []

    for query in queries:
        clean = re.sub(r"\s+", " ", str(query or "")).strip()
        lookup = clean.lower()

        if clean and lookup not in seen:
            seen.add(lookup)
            clean_queries.append(clean)

    return clean_queries


def _is_esearch_backend_error(message: str) -> bool:
    lower = str(message or "").lower()
    return (
        "search backend failed" in lower
        or "pmquerysrv" in lower
        or "address table is empty" in lower
    )


def _month_to_number(month_text: str) -> int:
    if not month_text:
        return 0

    value = str(month_text).strip().lower()

    if value.isdigit():
        number = int(value)
        return number if 1 <= number <= 12 else 0

    return MONTH_MAP.get(value[:3], MONTH_MAP.get(value, 0))


def _count_authors(authors: str) -> int:
    if not str(authors).strip():
        return 0

    return len([name for name in str(authors).split(";") if name.strip()])


def _first_keyword(keywords: str) -> str:
    for keyword in re.split(r";|\||,", str(keywords)):
        clean = keyword.strip()

        if clean:
            return clean

    return "Unknown"


def _deduplicate_pubmed(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df["_pmid_key"] = df["pmid"].replace("", pd.NA)
    df["_doi_key"] = df["doi"].str.lower().replace("", pd.NA)

    with_pmid = df[df["_pmid_key"].notna()].drop_duplicates(subset=["_pmid_key"], keep="first")
    without_pmid = df[df["_pmid_key"].isna()]
    df = pd.concat([with_pmid, without_pmid], ignore_index=True)

    with_doi = df[df["_doi_key"].notna()].drop_duplicates(subset=["_doi_key"], keep="first")
    without_doi = df[df["_doi_key"].isna()]
    df = pd.concat([with_doi, without_doi], ignore_index=True)

    return df.drop(columns=["_pmid_key", "_doi_key"], errors="ignore")


if __name__ == "__main__":
    warning = get_pubmed_config_warning()

    if warning:
        print(f"Warning: {warning}")

    demo_df = search_pubmed("alzheimer artificial intelligence", max_results=10)
    metadata = demo_df.attrs.get("pubmed_metadata", {})
    preview_cols = ["title", "year", "journal", "doi"]

    if demo_df.empty:
        print("PubMed service temporarily unavailable. Please retry later.")
        tried = metadata.get("tried_queries") or DEFAULT_ESEARCH_FALLBACK_QUERIES
        print("Tried fallback queries:")
        for item in tried:
            print(f"- {item}")
    else:
        print(demo_df[preview_cols].head(5).to_string(index=False))
