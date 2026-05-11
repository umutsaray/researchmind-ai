from __future__ import annotations

from datetime import datetime
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


KEEP_COLUMNS = [
    "pmid", "title", "abstract", "journal", "pub_year", "pub_month", "pub_month_num",
    "month_year", "authors_count", "country", "research_type", "keywords",
    "major_topic", "language", "open_access"
]

SYNONYMS = {
    "cancer": ["cancer", "tumor", "tumour", "neoplasm", "oncology", "carcinoma"],
    "ai": ["artificial intelligence", "machine learning", "deep learning", "neural network"],
    "diagnosis": ["diagnosis", "diagnostic", "detection", "screening", "classification"],
    "alzheimer": ["alzheimer", "dementia", "mild cognitive impairment"],
    "heart": ["heart", "cardiac", "cardiovascular"],
}


def expand_query_groups(query: str) -> list[list[str]]:
    query = str(query).lower().strip()

    phrase_matches = re.findall(r'"([^"]+)"', query)
    query_without_phrases = re.sub(r'"[^"]+"', "", query)

    groups = []

    for phrase in phrase_matches:
        groups.append([phrase.lower().strip()])

    for word in re.findall(r"\w+", query_without_phrases):
        if len(word) <= 2:
            continue

        if word in SYNONYMS:
            groups.append(SYNONYMS[word])
        else:
            groups.append([word])

    return groups


def read_csv_light(path: str | Path, nrows: Optional[int] = None) -> pd.DataFrame:
    path = Path(path)
    header = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [c for c in KEEP_COLUMNS if c in header]
    df = pd.read_csv(path, usecols=usecols, nrows=nrows, low_memory=False)
    return normalize_dataframe(df)


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in [
        "title", "abstract", "journal", "country", "research_type",
        "keywords", "major_topic", "language", "month_year"
    ]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)

    if "pub_year" in df.columns:
        df["pub_year"] = pd.to_numeric(df["pub_year"], errors="coerce").astype("Int64")

    if "pub_month_num" in df.columns:
        df["pub_month_num"] = pd.to_numeric(
            df["pub_month_num"], errors="coerce"
        ).fillna(0).astype(int)

    if "open_access" in df.columns:
        df["open_access"] = df["open_access"].astype(str)

    if "abstract" in df.columns:
        df["abstract_words_calc"] = df["abstract"].str.split().str.len()

    if "pub_year" in df.columns and "pub_month_num" in df.columns:
        df["date_index"] = (
            df["pub_year"].astype(str)
            + "-"
            + df["pub_month_num"].astype(str).str.zfill(2)
        )
        df.loc[df["pub_month_num"].eq(0), "date_index"] = df["pub_year"].astype(str) + "-Unknown"

    return df


def top_counts(df: pd.DataFrame, column: str, n: int = 20) -> pd.DataFrame:
    if column not in df.columns:
        return pd.DataFrame(columns=[column, "count"])

    out = (
        df[column]
        .replace("", "Unknown")
        .fillna("Unknown")
        .astype(str)
        .value_counts()
        .head(n)
        .reset_index()
    )
    out.columns = [column, "count"]
    return out


def publication_trend(df: pd.DataFrame) -> pd.DataFrame:
    if "pub_year" not in df.columns or df.empty:
        return pd.DataFrame()

    group_cols = ["pub_year"]

    if "pub_month_num" in df.columns:
        group_cols.append("pub_month_num")

    out = df.groupby(group_cols).size().reset_index(name="publication_count")

    if "pub_month_num" in out.columns:
        out = out.sort_values(["pub_year", "pub_month_num"])
        out["period"] = (
            out["pub_year"].astype(str)
            + "-"
            + out["pub_month_num"].astype(str).str.zfill(2)
        )
    else:
        out["period"] = out["pub_year"].astype(str)

    return out


def split_keywords(series: Iterable[str]) -> pd.Series:
    counter = {}

    for text in series:
        if not isinstance(text, str):
            continue

        parts = re.split(r";|\||,", text)

        for p in parts:
            k = p.strip().lower()
            if len(k) > 2 and k != "unknown":
                counter[k] = counter.get(k, 0) + 1

    return pd.Series(counter).sort_values(ascending=False)


def keyword_trend(df: pd.DataFrame, keyword: str) -> pd.DataFrame:
    if "abstract" not in df.columns or "pub_year" not in df.columns:
        return pd.DataFrame()

    groups = expand_query_groups(keyword)

    if not groups:
        return pd.DataFrame()

    search_text = df.get("title", pd.Series("", index=df.index)).astype(str).str.lower()
    search_text += " " + df.get("abstract", pd.Series("", index=df.index)).astype(str).str.lower()
    search_text += " " + df.get("keywords", pd.Series("", index=df.index)).astype(str).str.lower()
    search_text += " " + df.get("major_topic", pd.Series("", index=df.index)).astype(str).str.lower()

    mask = pd.Series(True, index=df.index)

    for group in groups:
        pattern = "|".join(re.escape(term.lower()) for term in group)
        group_mask = search_text.str.contains(pattern, na=False, regex=True)
        mask = mask & group_mask

    return publication_trend(df.loc[mask])


def semantic_search_tfidf(df: pd.DataFrame, query: str, top_k: int = 10) -> pd.DataFrame:
    result_cols = [
        c for c in [
            "pmid", "title", "journal", "pub_year", "country",
            "research_type", "major_topic", "abstract"
        ]
        if c in df.columns
    ]

    if df.empty:
        return pd.DataFrame(columns=["similarity_score"] + result_cols)

    text = (
        df.get("title", pd.Series("", index=df.index)).astype(str)
        + " "
        + df.get("abstract", pd.Series("", index=df.index)).astype(str)
    ).fillna("")

    max_features = min(50000, max(1000, len(df) * 5))
    min_df = 1 if len(df) < 20 else 2

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=min_df,
    )

    try:
        X = vectorizer.fit_transform(text)
    except ValueError:
        return pd.DataFrame(columns=["similarity_score"] + result_cols)

    qv = vectorizer.transform([query])
    scores = cosine_similarity(qv, X).ravel()

    idx = np.argsort(scores)[::-1][:top_k]

    out = df.iloc[idx][result_cols].copy()
    out.insert(0, "similarity_score", scores[idx])

    return out


def research_gap_score(df: pd.DataFrame, query: str) -> dict:
    trend = keyword_trend(df, query)

    if trend.empty:
        return {
            "query": query,
            "total_records": 0,
            "growth_rate": "-",
            "gap_score": 85,
            "interpretation": "Very low publication volume; possible niche gap, verify manually."
        }

    total = int(trend["publication_count"].sum())

    early = trend.head(max(1, len(trend) // 2))["publication_count"].mean()
    late = trend.tail(max(1, len(trend) // 2))["publication_count"].mean()

    growth = 0 if early == 0 else (late - early) / early

    saturation_penalty = min(total / 5000, 1.0)

    growth_score = min(max(growth, 0), 1.0)
    density_score = 1 - saturation_penalty

    raw = (0.60 * density_score + 0.40 * growth_score) * 100
    score = float(np.clip(raw, 0, 100))

    if total < 20:
        note = "Low volume: promising gap, but evidence base is small."
    elif growth > 0.5 and total < 1000:
        note = "Emerging topic: strong gap/opportunity candidate."
    elif total > 3000:
        note = "High saturation: gap should be narrowed to a subtopic."
    else:
        note = "Moderate opportunity: refine query with disease, method, or population."

    return {
        "query": query,
        "total_records": total,
        "growth_rate": round(float(growth), 3),
        "gap_score": round(score, 2),
        "interpretation": note,
    }


def suggest_research_opportunities(df: pd.DataFrame, query: str, top_n: int = 8) -> pd.DataFrame:
    if "major_topic" not in df.columns:
        return pd.DataFrame()

    stop_topics = {
        "unknown", "humans", "human", "male", "female", "aged", "middle aged",
        "young adult", "adult", "animals", "animal", "child", "adolescent",
        "infant", "rats", "mice", "retrospective studies", "prospective studies",
        "surveys and questionnaires", "models", "computer-assisted",
        "animal distribution"
    }

    topics = df["major_topic"].dropna().astype(str).str.strip()
    topics = topics[~topics.str.lower().isin(stop_topics)]
    top_topics = topics.value_counts().head(12).index.tolist()

    rows = []

    for topic in top_topics:
        combined_query = f'{query} "{topic}"'
        gap = research_gap_score(df, combined_query)

        total = gap.get("total_records", 0)
        score = gap.get("gap_score", 0)
        growth = gap.get("growth_rate", "-")

        if total == 0:
            opportunity_type = "Niche / underexplored"
        elif total < 20:
            opportunity_type = "Very low-volume opportunity"
        elif total < 100:
            opportunity_type = "Low-volume opportunity"
        elif total < 500:
            opportunity_type = "Emerging opportunity"
        else:
            opportunity_type = "Crowded field"

        rows.append({
            "suggested_topic": combined_query,
            "base_topic": topic,
            "matched_records": total,
            "growth_rate": growth,
            "gap_score": score,
            "opportunity_type": opportunity_type,
            "recommendation": gap.get("interpretation", "")
        })

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    out = out.sort_values(
        by=["gap_score", "matched_records"],
        ascending=[False, True]
    ).head(top_n)

    return out


def _openalex_params(api_key: str = "") -> dict:
    params = {}
    key_value = str(api_key).strip()

    if key_value:
        if "@" in key_value:
            params["mailto"] = key_value
        else:
            params["api_key"] = key_value

    return params


def openalex_count_by_year(query: str, year: int, api_key: str = "") -> int:
    url = "https://api.openalex.org/works"

    params = {
        "search": query,
        "filter": f"publication_year:{year}",
        "per_page": 1,
    }

    params.update(_openalex_params(api_key))

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    return int(data.get("meta", {}).get("count", 0))


def search_openalex_works(query: str, api_key: str = "", per_page: int = 10) -> pd.DataFrame:
    url = "https://api.openalex.org/works"

    params = {
        "search": query,
        "per_page": per_page,
        "sort": "cited_by_count:desc",
    }

    params.update(_openalex_params(api_key))

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    rows = []

    for item in data.get("results", []):
        primary_location = item.get("primary_location") or {}
        source = primary_location.get("source") or {}
        primary_topic = item.get("primary_topic") or {}

        rows.append({
            "title": item.get("display_name"),
            "publication_year": item.get("publication_year"),
            "cited_by_count": item.get("cited_by_count"),
            "doi": item.get("doi"),
            "openalex_id": item.get("id"),
            "type": item.get("type"),
            "source": source.get("display_name"),
            "primary_topic": primary_topic.get("display_name"),
            "open_access": (item.get("open_access") or {}).get("is_oa"),
        })

    return pd.DataFrame(rows)


def openalex_gap_analysis(query: str, api_key: str = "", per_page: int = 20, years_back: int = 5) -> dict:
    current_year = datetime.now().year
    years = list(range(current_year - years_back + 1, current_year + 1))

    trend_rows = []

    for year in years:
        count = openalex_count_by_year(query, year, api_key=api_key)
        trend_rows.append({
            "period": str(year),
            "publication_year": year,
            "publication_count": count,
        })

    trend = pd.DataFrame(trend_rows)
    total = int(trend["publication_count"].sum())

    if total == 0:
        results = search_openalex_works(query=query, api_key=api_key, per_page=per_page)
        return {
            "query": query,
            "total_records": 0,
            "growth_rate": "-",
            "gap_score": 85,
            "interpretation": "Very low OpenAlex volume in the last 5 years; possible niche gap, verify manually.",
            "results": results,
            "trend": trend,
        }

    early = trend.head(max(1, len(trend) // 2))["publication_count"].mean()
    late = trend.tail(max(1, len(trend) // 2))["publication_count"].mean()
    growth = 0 if early == 0 else (late - early) / early

    if total < 50:
        density_score = 1.0
    elif total < 200:
        density_score = 0.75
    elif total < 1000:
        density_score = 0.45
    elif total < 5000:
        density_score = 0.20
    else:
        density_score = 0.05

    growth_score = min(max(growth, 0), 1.0)

    raw_score = (0.65 * density_score + 0.35 * growth_score) * 100
    score = float(np.clip(raw_score, 0, 100))

    if total < 50:
        note = "Low recent publication volume: possible underexplored topic."
    elif total < 200 and growth > 0:
        note = "Emerging opportunity: recent volume is still manageable and trend is growing."
    elif total >= 1000:
        note = "High recent publication volume: narrow the topic further."
    else:
        note = "Moderate opportunity: refine by method, disease, dataset, or population."

    results = search_openalex_works(query=query, api_key=api_key, per_page=per_page)

    return {
        "query": query,
        "total_records": total,
        "growth_rate": round(float(growth), 3),
        "gap_score": round(score, 2),
        "interpretation": note,
        "results": results,
        "trend": trend,
    }

def generate_ai_research_topic_suggestions(query: str, api_key: str = "", years_back: int = 5) -> pd.DataFrame:
    """
    Generates AI-like research topic suggestions using OpenAlex last-N-year volume,
    growth, and narrowing strategies.
    """

    base_ideas = [
        "explainable AI",
        "multimodal learning",
        "early diagnosis",
        "small dataset learning",
        "federated learning",
        "clinical decision support",
        "medical imaging",
        "MRI",
        "PET",
        "EEG",
        "biomarker prediction",
        "risk prediction",
        "deep learning",
        "transformer model",
    ]

    rows = []

    for idea in base_ideas:
        suggested_query = f"{query} {idea}"
        gap = openalex_gap_analysis(
            query=suggested_query,
            api_key=api_key,
            per_page=10,
            years_back=years_back,
        )

        rows.append({
            "suggested_research_topic": suggested_query,
            "matched_records_last_5_years": gap.get("total_records", 0),
            "growth_rate": gap.get("growth_rate", "-"),
            "gap_score": gap.get("gap_score", "-"),
            "recommendation": gap.get("interpretation", ""),
        })

    out = pd.DataFrame(rows)

    if out.empty:
        return out

    return out.sort_values(
        by=["gap_score", "matched_records_last_5_years"],
        ascending=[False, True]
    ).head(10)
