from __future__ import annotations

from datetime import datetime
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import time
import traceback
import unicodedata

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config_utils import get_config_bool, get_config_value
from pubmed_client import (
    PubMedClient,
    PubMedClientError,
    PubMedConfig,
    get_pubmed_config,
    normalize_pubmed_to_researchmind_schema,
)
from trend_engine import (
    generate_ai_research_topic_suggestions,
    keyword_trend,
    openalex_gap_analysis,
    publication_trend,
    read_csv_light,
    research_gap_score,
    semantic_search_tfidf,
    split_keywords,
    suggest_research_opportunities,
    top_counts,
)


DEFAULT_LOCAL_QUERY = "artificial intelligence cancer diagnosis"
DEFAULT_LIVE_QUERY = "alzheimer artificial intelligence"
DEFAULT_LIVE_RESULT_LIMIT = 100
DEMO_LIVE_RESULT_LIMIT = 50
OUTPUTS_DIR = Path("outputs")
DEMO_LOGS_DIR = Path("demo_logs")
DEMO_CACHE_DIR = Path("demo_cache")

DATA_SOURCE_LABELS = {
    "Local CSV": "Yerel Veri Seti",
    "OpenAlex Live": "OpenAlex Canlı Veri",
    "PubMed Live": "PubMed Canlı Veri",
    "Hybrid: OpenAlex + PubMed": "Hibrit Analiz",
}

HEALTHCARE_DOMAIN = "Healthcare & Biomedical Sciences"
ENGINEERING_DOMAIN = "Engineering & Applied Technologies"
ACTIVE_RESEARCH_DOMAINS = [HEALTHCARE_DOMAIN, ENGINEERING_DOMAIN]
DOMAIN_SUPPORT_WARNING = (
    "Bu demo sürümünde yalnızca sağlık/biyomedikal ve mühendislik alanları desteklenmektedir. "
    "Lütfen konunuzu bu alanlardan biriyle ilişkilendirerek tekrar deneyin."
)

ENGINEERING_KEYWORDS = [
    "engineering", "seismic", "earthquake", "structural", "civil engineering",
    "bridge", "building", "construction", "geotechnical", "infrastructure",
    "structural health monitoring", "shm", "cfd", "finite element",
    "thermodynamics", "heat transfer", "robotics", "uav", "drone",
    "wireless communication", "signal processing", "iot", "embedded systems",
    "microcontroller", "power systems", "renewable energy", "machine learning",
    "deep learning", "computer vision", "cybersecurity", "blockchain",
    "materials", "nanomaterials", "composite materials", "3d printing",
    "manufacturing", "optimization", "simulation", "digital twin", "control systems",
]

HEALTHCARE_KEYWORDS = [
    "healthcare", "medical", "clinical", "disease", "patient", "mri", "ct",
    "eeg", "ecg", "alzheimer", "cancer", "diagnosis", "treatment", "hospital",
    "biosensor", "radiology", "genomics", "neurology", "cardiology",
    "public health", "medical imaging", "wearable health", "digital health",
]

BLOCKED_KEYWORDS = [
    "tax", "taxation", "fiscal", "economics", "public expenditure", "inflation",
    "banking", "monetary policy", "stock market", "cryptocurrency trading",
    "accounting", "audit", "tax compliance", "fiscal sustainability",
    "government spending", "budget deficit", "public finance", "fiscal risk",
    "public spending", "vergi uyumu", "kamu harcamalar", "mali surdur",
]

UNSUPPORTED_DOMAIN_TERMS = set(BLOCKED_KEYWORDS)
HEALTHCARE_TERMS = set(HEALTHCARE_KEYWORDS)
ENGINEERING_TERMS = set(ENGINEERING_KEYWORDS)

HEALTHCARE_BLACKLIST = {
    "tax compliance", "fiscal sustainability", "public expenditure", "macroeconomic",
    "government spending", "budget deficit", "public finance", "fiscal risk",
    "public spending", "fiscal", "tax", "vergi uyumu", "kamu harcamalar", "mali surdur",
}

ENGINEERING_BIOMED_BLACKLIST = {
    "alzheimer", "dementia", "cancer", "patient cohort", "clinical validation",
    "mri", "pet", "biomarker", "diagnosis", "disease",
}

ENGINEERING_HEALTH_HYBRID_TERMS = {
    "biomedical engineering", "medical device", "biomedical signal processing",
    "medical imaging device", "health technology", "wearable health",
}

STOP_TOPICS = {
    "unknown", "humans", "human", "male", "female", "aged", "middle aged",
    "young adult", "adult", "animals", "animal", "retrospective studies",
    "prospective studies", "china", "united states", "child", "adolescent",
    "infant", "rats", "mice", "surveys and questionnaires", "models",
    "computer-assisted", "covid-19", "sars-cov-2", "coronavirus",
    "animal distribution",
}

DOMAIN_ONTOLOGY = {
    "medical_imaging": {
        "label": "Medical Imaging",
        "terms": ["mri", "ct", "pet", "mammography", "histopathology", "radiology", "neuroimaging", "medical imaging", "x ray", "ultrasound"],
    },
    "signal_processing": {
        "label": "Signal Processing",
        "terms": ["eeg", "ecg", "emg", "biosignal", "waveform", "spectral", "wavelet"],
    },
    "speech": {
        "label": "Speech / Acoustic",
        "terms": ["voice", "speech", "acoustic", "audio"],
    },
    "movement": {
        "label": "Movement Analysis",
        "terms": ["gait", "motion", "activity recognition", "movement"],
    },
    "oncology": {
        "label": "Oncology",
        "terms": ["breast cancer", "lung cancer", "cancer", "tumor", "tumour", "histopathology", "mammography", "biopsy"],
    },
    "neurodegenerative": {
        "label": "Neurology / Neurodegenerative",
        "terms": ["alzheimer", "dementia", "parkinson", "neurodegeneration", "cognition", "neuroimaging"],
    },
    "mental_health": {
        "label": "Mental Health",
        "terms": ["depression", "anxiety", "psychiatric", "mental health"],
    },
    "neurodevelopmental": {
        "label": "Neurodevelopmental AI",
        "terms": ["autism", "asd", "neurodevelopmental", "eye tracking", "gaze", "developmental screening"],
    },
    "healthcare_security": {
        "label": "Healthcare Security",
        "terms": ["blockchain", "security", "privacy", "interoperability", "smart contract", "healthcare data"],
    },
    "sports_medicine": {
        "label": "Sports Medicine / Sports Analytics",
        "terms": [
            "sports medicine", "sports analytics", "athlete monitoring", "football", "soccer",
            "injury risk", "injury prediction", "player workload", "performance analytics",
            "training load", "match statistics", "biomechanical data",
        ],
    },
}

CONCEPT_PATTERNS = {
    "disease": {
        "breast cancer": ["breast cancer", "breast tumor", "mammography"],
        "alzheimer": ["alzheimer", "dementia"],
        "parkinson": ["parkinson"],
        "depression": ["depression"],
        "autism": ["autism", "asd", "autism spectrum"],
        "sports injury": ["injury risk", "injury prediction", "injury", "sports injury"],
        "cancer": ["cancer", "tumor", "tumour"],
    },
    "modality": {
        "medical imaging": ["medical imaging", "mri", "ct", "pet", "mammography", "histopathology", "radiology", "neuroimaging"],
        "mri": ["mri", "magnetic resonance"],
        "pet": ["pet"],
        "eeg": ["eeg", "electroencephalography"],
        "eye tracking": ["eye tracking", "gaze", "ocular"],
        "biosignal": ["ecg", "emg", "biosignal", "waveform"],
        "blockchain records": ["blockchain", "healthcare data", "ehr", "medical records"],
        "wearable sensors": ["wearable", "wearable sensors", "sensor", "accelerometer", "heart rate"],
        "gps tracking": ["gps", "gps tracking", "tracking"],
        "training load": ["training load", "player workload", "workload"],
        "match statistics": ["match statistics", "match stats", "performance analytics"],
        "biomechanical data": ["biomechanical", "biomechanical data", "movement data", "video tracking"],
    },
    "method": {
        "cnn": ["cnn", "convolutional", "efficientnet", "resnet", "densenet"],
        "deep learning": ["deep learning", "neural network"],
        "vision transformer": ["vision transformer", "vision transformers", "vit", "transformer", "transformers"],
        "explainable ai": ["explainable", "xai", "shap", "lime", "grad cam", "grad-cam"],
        "federated learning": ["federated"],
        "multimodal learning": ["multimodal", "fusion"],
        "blockchain": ["blockchain", "smart contract"],
        "time-series deep learning": ["time series", "time-series", "lstm", "gru", "temporal transformer", "workload modeling"],
    },
    "task": {
        "classification": ["classification", "classify"],
        "diagnosis": ["diagnosis", "detection", "screening"],
        "analysis": ["analysis", "assessment"],
        "injury risk prediction": ["injury risk", "injury prediction", "risk assessment", "risk scoring", "risk stratification"],
        "security": ["security", "privacy", "authentication", "interoperability"],
    },
    "clinical_domain": {
        "oncology": ["breast cancer", "lung cancer", "cancer", "tumor", "mammography", "histopathology"],
        "neurology": ["alzheimer", "dementia", "parkinson", "mri", "pet", "neuroimaging"],
        "mental health": ["depression", "anxiety", "psychiatric"],
        "neurodevelopmental": ["autism", "asd", "neurodevelopmental", "eye tracking", "developmental"],
        "healthcare security": ["blockchain", "security", "privacy", "interoperability"],
        "sports medicine": ["football", "soccer", "athlete", "player", "injury", "sports", "training load", "match statistics"],
    },
    "modifiers": {
        "explainability": ["explainable", "xai", "shap", "lime", "grad cam", "grad-cam"],
        "privacy": ["privacy", "federated", "blockchain", "security"],
        "clinical validation": ["clinical validation", "multi center", "multi-center", "clinical decision support"],
    },
}

LEAKAGE_TERMS = {
    "covid", "sars", "parkinson", "eeg", "speech", "voice", "gait",
    "emotion recognition", "activity recognition",
}


st.set_page_config(page_title="ResearchMind AI", layout="wide")


def inject_product_styles() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            min-width: 360px;
            max-width: 420px;
        }
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: 1.05rem;
        }
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            margin-top: 0.35rem;
            margin-bottom: 0.55rem;
            letter-spacing: 0;
        }
        [data-testid="stSidebar"] .stButton > button {
            min-height: 3rem;
            border-radius: 8px;
            font-weight: 700;
            letter-spacing: 0;
        }
        .rm-hero {
            padding: 1.7rem 1.8rem;
            border: 1px solid rgba(255,255,255,0.11);
            border-radius: 12px;
            background: linear-gradient(135deg, rgba(18,28,45,0.96), rgba(26,45,54,0.92));
            margin-bottom: 1.2rem;
        }
        .rm-hero h1 {
            margin: 0;
            font-size: 2.55rem;
            line-height: 1.05;
            letter-spacing: 0;
        }
        .rm-hero h2 {
            margin: 0.4rem 0 1rem 0;
            font-size: 1.05rem;
            font-weight: 600;
            color: rgba(255,255,255,0.78);
            letter-spacing: 0;
        }
        .rm-hero p {
            margin: 1rem 0 0 0;
            max-width: 860px;
            color: rgba(255,255,255,0.82);
            font-size: 1rem;
        }
        .rm-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
        }
        .rm-badge {
            padding: 0.42rem 0.7rem;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.14);
            background: rgba(255,255,255,0.08);
            font-weight: 650;
            font-size: 0.88rem;
        }
        .rm-card {
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 1rem 1.05rem;
            background: rgba(255,255,255,0.045);
            min-height: 7.2rem;
        }
        .rm-card-label {
            color: rgba(255,255,255,0.66);
            font-size: 0.86rem;
            margin-bottom: 0.45rem;
        }
        .rm-card-value {
            font-size: 1.65rem;
            font-weight: 750;
            line-height: 1.15;
        }
        .rm-card-note {
            color: rgba(255,255,255,0.72);
            margin-top: 0.45rem;
            font-size: 0.86rem;
        }
        .rm-status {
            display: inline-block;
            margin-top: 0.55rem;
            padding: 0.32rem 0.58rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.8rem;
        }
        .rm-status-low { background: rgba(239,68,68,0.18); color: #fca5a5; border: 1px solid rgba(239,68,68,0.35); }
        .rm-status-mid { background: rgba(245,158,11,0.18); color: #fcd34d; border: 1px solid rgba(245,158,11,0.35); }
        .rm-status-high { background: rgba(34,197,94,0.16); color: #86efac; border: 1px solid rgba(34,197,94,0.34); }
        .rm-insight {
            border: 1px solid rgba(56,189,248,0.28);
            box-shadow: 0 0 28px rgba(56,189,248,0.12);
            border-radius: 12px;
            background: linear-gradient(135deg, rgba(56,189,248,0.12), rgba(34,197,94,0.055));
            padding: 1.1rem 1.2rem;
            margin: 1rem 0 0.6rem 0;
            color: rgba(255,255,255,0.86);
        }
        .rm-insight-title {
            font-weight: 800;
            font-size: 1.02rem;
            margin-bottom: 0.45rem;
        }
        .rm-big-badge {
            display: inline-block;
            padding: 0.55rem 0.8rem;
            border-radius: 999px;
            font-weight: 850;
            font-size: 0.9rem;
            letter-spacing: 0;
            margin-top: 0.65rem;
        }
        .rm-opportunity {
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 1rem;
            background: linear-gradient(180deg, rgba(255,255,255,0.065), rgba(255,255,255,0.035));
            box-shadow: 0 10px 26px rgba(0,0,0,0.18);
            height: 100%;
        }
        .rm-opportunity h4 {
            margin: 0 0 0.75rem 0;
            font-size: 1rem;
            line-height: 1.3;
            letter-spacing: 0;
        }
        .rm-muted {
            color: rgba(255,255,255,0.66);
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(results: dict | None = None) -> None:
    diagnostics = (results or {}).get("diagnostics", {})
    distribution = (results or {}).get("source_distribution", {})
    has_pubmed = distribution.get("PubMed", 0) > 0
    has_openalex = distribution.get("OpenAlex", 0) > 0
    pubmed_failed = diagnostics.get("pubmed_called") and diagnostics.get("pubmed_error")

    if pubmed_failed and has_openalex:
        status_line = f"⚠ PubMed unavailable, OpenAlex active · PubMed: {distribution.get('PubMed', 0)} · OpenAlex: {distribution.get('OpenAlex', 0)}"
    elif has_pubmed and has_openalex:
        status_line = f"✔ Hybrid Intelligence Active · PubMed: {distribution.get('PubMed', 0)} · OpenAlex: {distribution.get('OpenAlex', 0)}"
    elif has_pubmed:
        status_line = f"✔ PubMed Connected · PubMed: {distribution.get('PubMed', 0)}"
    elif has_openalex:
        status_line = f"✔ OpenAlex Connected · OpenAlex: {distribution.get('OpenAlex', 0)}"
    else:
        status_line = "✔ PubMed Ready &nbsp;&nbsp; ✔ OpenAlex Ready &nbsp;&nbsp; ✔ Hybrid Intelligence Ready"

    st.markdown(
        f"""
        <div class="rm-hero">
            <h1>ResearchMind AI</h1>
            <h2>AI-Powered Research Intelligence Platform</h2>
            <div class="rm-badges">
                <span class="rm-badge">Trend Analysis</span>
                <span class="rm-badge">Research Gap Detection</span>
                <span class="rm-badge">AI Topic Recommendation</span>
            </div>
            <div class="rm-card-note" style="margin-top: 0.9rem;">{status_line}</div>
            <p>Araştırma konusunu gir, analiz dönemini seç ve tek tıkla trend, fırsat ve Research Gap Score sonuçlarını üret.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_plotly_chart(fig):
    fig.update_traces(line={"width": 3}, marker={"size": 8}, selector={"type": "scatter"})
    fig.update_traces(marker_line_width=0, selector={"type": "bar"})
    fig.update_layout(
        template="plotly_dark",
        title={"font": {"size": 22}},
        font={"size": 14},
        xaxis={"gridcolor": "rgba(255,255,255,0.08)", "title_font": {"size": 15}, "tickfont": {"size": 13}},
        yaxis={"gridcolor": "rgba(255,255,255,0.08)", "title_font": {"size": 15}, "tickfont": {"size": 13}},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin={"l": 18, "r": 18, "t": 60, "b": 35},
    )
    return fig


def localize_text(value) -> str:
    text = str(value or "")
    replacements = {
        "Low recent publication volume: possible underexplored topic.": "Son yıllardaki yayın hacmi düşük. Konu keşfedilmemiş bir araştırma fırsatı içerebilir.",
        "Emerging opportunity: recent volume is still manageable and trend is growing.": "Yükselen fırsat: yayın hacmi hâlâ yönetilebilir ve trend büyüme gösteriyor.",
        "High recent publication volume: narrow the topic further.": "Son yıllardaki yayın hacmi yüksek. Daha güçlü bir fırsat için konuyu daraltmanız önerilir.",
        "Moderate opportunity: refine by method, disease, dataset, or population.": "Orta düzey fırsat: yöntemi, hastalık alanını, veri setini veya hedef popülasyonu daraltın.",
        "Very low OpenAlex volume in the last 5 years; possible niche gap, verify manually.": "Son 5 yılda OpenAlex hacmi çok düşük. Niş bir boşluk olabilir; manuel doğrulama önerilir.",
        "Low volume: promising gap, but evidence base is small.": "Düşük yayın hacmi var. Fırsat potansiyeli taşıyor ancak kanıt tabanı sınırlı.",
        "Emerging topic: strong gap/opportunity candidate.": "Yükselen konu: güçlü araştırma boşluğu ve fırsat adayı.",
        "High saturation: gap should be narrowed to a subtopic.": "Alan doygun görünüyor. Daha net bir boşluk için alt konuya daraltılmalı.",
        "Moderate opportunity: refine query with disease, method, or population.": "Orta düzey fırsat: hastalık, yöntem veya popülasyonla sorguyu daraltın.",
        "Semantic matching found substantial related literature; this is a competitive area and should be narrowed.": "Semantic matching anlamlı büyüklükte ilişkili literatür buldu; bu rekabetçi bir alan ve daraltılmalı.",
        "Semantic matching found a meaningful related corpus; avoid niche-gap interpretation and refine the subtopic.": "Semantic matching anlamlı bir ilişkili yayın kümesi buldu; niş boşluk yorumu yapılmamalı, alt konu daraltılmalı.",
        "Semantic matching found a small but growing related corpus; potential opportunity, verify manually.": "Semantic matching küçük ama büyüyen bir yayın kümesi buldu; fırsat olabilir, manuel doğrulama önerilir.",
        "Semantic matching found limited related literature; refine query and validate coverage.": "Semantic matching sınırlı ilişkili literatür buldu; sorguyu iyileştirip kaynak kapsamını doğrulayın.",
        "Semantic matching found no reliable matches; broaden the query or verify source coverage.": "Semantic matching güvenilir eşleşme bulamadı; sorguyu genişletin veya kaynak kapsamını doğrulayın.",
    }
    return replacements.get(text, text)


def parse_numeric(value, default: float = 0.0) -> float:
    try:
        if value in ("", "-", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


SEMANTIC_CONCEPTS = {
    "explainable ai": {
        "terms": ["explainable ai", "xai", "interpretable", "explainability", "shap", "lime", "grad-cam", "grad cam"],
        "weight": 1.25,
    },
    "vision transformer": {
        "terms": ["vision transformer", "vit", "transformer", "transformer-based", "vision transformers"],
        "weight": 1.2,
    },
    "alzheimer": {
        "terms": ["alzheimer", "dementia", "mild cognitive impairment", "neurodegenerative"],
        "weight": 1.25,
    },
    "mri": {
        "terms": ["mri", "magnetic resonance", "neuroimaging", "brain imaging", "structural imaging"],
        "weight": 1.15,
    },
    "medical imaging": {
        "terms": ["medical imaging", "imaging", "radiology", "pet", "ct", "fmri", "brain scan"],
        "weight": 0.9,
    },
    "diagnosis": {
        "terms": ["diagnosis", "diagnostic", "detection", "classification", "screening", "prediction"],
        "weight": 0.85,
    },
    "deep learning": {
        "terms": ["deep learning", "neural network", "cnn", "machine learning", "artificial intelligence", "ai"],
        "weight": 0.75,
    },
    "clinical validation": {
        "terms": ["clinical validation", "clinical decision support", "clinical", "patient", "cohort"],
        "weight": 0.7,
    },
}


def semantic_query_concepts(query: str) -> list[dict]:
    query_key = normalize_topic_key(preprocess_research_query(query))
    concepts = []

    for name, spec in SEMANTIC_CONCEPTS.items():
        terms = spec["terms"]
        if any(normalize_topic_key(term) in query_key for term in [name, *terms]):
            concepts.append({"name": name, "terms": terms, "weight": spec["weight"]})

    raw_tokens = [
        token for token in query_key.split()
        if len(token) > 2 and token not in {"for", "and", "with", "based", "analysis", "study"}
    ]
    existing_terms = " ".join(term for concept in concepts for term in concept["terms"])

    for token in raw_tokens:
        if token not in existing_terms and token not in {concept["name"] for concept in concepts}:
            concepts.append({"name": token, "terms": [token], "weight": 0.45})

    return concepts


def pubmed_fallback_queries(query: str) -> list[str]:
    key = normalize_topic_key(query)
    concepts = {concept["name"] for concept in semantic_query_concepts(query)}
    queries = [preprocess_research_query(query)]

    has_alzheimer = "alzheimer" in key
    has_mri = "mri" in key or "magnetic resonance" in key or "neuroimaging" in key
    has_explainable = "explainable" in key or "xai" in key or "explainable ai" in concepts
    has_transformer = "transformer" in key or "vision transformer" in key or "vit" in key
    has_ai = "artificial intelligence" in key or " ai " in f" {key} " or "deep learning" in key

    if has_alzheimer and has_mri and (has_explainable or has_transformer or has_ai):
        queries.extend([
            "Explainable AI Alzheimer MRI",
            "Alzheimer MRI artificial intelligence",
            "Alzheimer disease MRI deep learning",
            "Alzheimer disease artificial intelligence",
        ])
    elif "alzheimer" in concepts and ("deep learning" in concepts or "explainable ai" in concepts):
        queries.extend([
            "Alzheimer disease artificial intelligence",
            "Alzheimer disease deep learning",
            "Alzheimer diagnosis machine learning",
        ])
    elif "mri" in concepts and "deep learning" in concepts:
        queries.extend([
            "MRI artificial intelligence",
            "medical imaging deep learning",
            "MRI deep learning diagnosis",
        ])

    if "vision transformer" in concepts:
        queries.append("vision transformer medical imaging")

    seen = set()
    clean_queries = []
    for item in queries:
        clean = preprocess_research_query(item)
        lookup = clean.lower()
        if clean and lookup not in seen:
            seen.add(lookup)
            clean_queries.append(clean)

    return clean_queries


def is_transient_pubmed_error(message: str) -> bool:
    lower = str(message).lower()
    return any(
        marker in lower
        for marker in [
            "search backend failed",
            "pmquerysrv",
            "address table is empty",
            "temporarily unavailable",
            "timed out",
            "connection",
            "backend",
        ]
    )


def semantic_search_text(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype="object")

    text = pd.Series("", index=df.index, dtype="object")
    for col in ["title", "abstract", "keywords", "major_topic", "research_type", "journal"]:
        if col in df.columns:
            text = text + " " + df[col].fillna("").astype(str)
    return text.str.lower().map(normalize_topic_key)


def semantic_match_scores(df: pd.DataFrame, query: str) -> pd.Series:
    if df.empty:
        return pd.Series(dtype="float64")

    concepts = semantic_query_concepts(query)
    text = semantic_search_text(df)
    max_weight = sum(concept["weight"] for concept in concepts) or 1.0
    overlap = pd.Series(0.0, index=df.index)

    for concept in concepts:
        pattern_terms = [normalize_topic_key(term) for term in concept["terms"] if normalize_topic_key(term)]
        if not pattern_terms:
            continue
        pattern = "|".join(re.escape(term) for term in pattern_terms)
        overlap = overlap + text.str.contains(pattern, regex=True, na=False).astype(float) * concept["weight"]

    overlap = overlap / max_weight

    try:
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        corpus = text.fillna("").tolist()
        matrix = vectorizer.fit_transform(corpus + [preprocess_research_query(query)])
        tfidf_scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
        tfidf = pd.Series(tfidf_scores, index=df.index)
    except ValueError:
        tfidf = pd.Series(0.0, index=df.index)

    return (0.72 * overlap + 0.28 * tfidf).clip(0, 1)


def semantic_threshold(query: str) -> float:
    concept_count = len(semantic_query_concepts(query))
    token_count = len(preprocess_research_query(query).split())

    if concept_count >= 5 or token_count >= 7:
        return 0.14
    if concept_count >= 3 or token_count >= 4:
        return 0.20
    return 0.28


def semantic_match_mask(df: pd.DataFrame, query: str) -> pd.Series:
    scores = semantic_match_scores(df, query)
    if scores.empty:
        return pd.Series(False, index=df.index)
    return scores >= semantic_threshold(query)


def semantic_query_trend(df: pd.DataFrame, query: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    mask = semantic_match_mask(df, query)
    return publication_trend(df.loc[mask])


def semantic_research_gap_score(df: pd.DataFrame, query: str, strict_gap: dict | None = None) -> dict:
    trend = semantic_query_trend(df, query)
    total = int(trend["publication_count"].sum()) if not trend.empty else 0
    strict_total = int(parse_numeric((strict_gap or {}).get("total_records")))

    if total == 0:
        return {
            "query": query,
            "total_records": 0,
            "strict_matched_records": strict_total,
            "growth_rate": "-",
            "gap_score": 75,
            "interpretation": "Semantic matching found no reliable matches; broaden the query or verify source coverage.",
            "matching_method": "semantic_topic_tfidf",
        }

    early = trend.head(max(1, len(trend) // 2))["publication_count"].mean()
    late = trend.tail(max(1, len(trend) // 2))["publication_count"].mean()
    growth = 0 if early == 0 else (late - early) / early

    density_penalty = min(total / 2500, 1.0)
    growth_score = min(max(growth, 0), 1.0)
    raw = (0.55 * (1 - density_penalty) + 0.45 * growth_score) * 100

    if total > 30:
        raw = min(raw, 82)

    score = round(float(max(0, min(raw, 100))), 2)

    if total > 500:
        note = "Semantic matching found substantial related literature; this is a competitive area and should be narrowed."
    elif total > 30:
        note = "Semantic matching found a meaningful related corpus; avoid niche-gap interpretation and refine the subtopic."
    elif growth > 0:
        note = "Semantic matching found a small but growing related corpus; potential opportunity, verify manually."
    else:
        note = "Semantic matching found limited related literature; refine query and validate coverage."

    return {
        "query": query,
        "total_records": total,
        "strict_matched_records": strict_total,
        "growth_rate": round(float(growth), 3),
        "gap_score": score,
        "interpretation": note,
        "matching_method": "semantic_topic_tfidf",
    }


def classify_gap_score(score) -> tuple[str, str]:
    numeric = parse_numeric(score)
    if numeric < 30:
        return "Düşük fırsat / yüksek rekabet", "low"
    if numeric < 70:
        return "Orta fırsat / konu daraltılmalı", "mid"
    return "Yüksek fırsat / güçlü araştırma potansiyeli", "high"


def opportunity_status(score) -> tuple[str, str]:
    numeric = parse_numeric(score)
    if numeric < 30:
        return "🔴 SATURATED / HIGH COMPETITION", "low"
    if numeric < 60:
        return "🟡 COMPETITIVE BUT REFINEABLE", "mid"
    if numeric < 80:
        return "🟢 EMERGING OPPORTUNITY", "high"
    return "🟢 STRONG STRATEGIC OPPORTUNITY", "high"


def strategic_level(score) -> tuple[str, str]:
    numeric = parse_numeric(score)
    if numeric < 30:
        return "Doygun / yüksek rekabet", "low"
    if numeric < 60:
        return "Rekabetçi fakat daraltılabilir", "mid"
    if numeric < 80:
        return "Yükselen araştırma fırsatı", "high"
    return "Güçlü stratejik araştırma fırsatı", "high"


def opportunity_trend_status(score, growth) -> tuple[str, str]:
    numeric_score = parse_numeric(score)
    numeric_growth = parse_numeric(growth)
    if numeric_score >= 70 or numeric_growth > 0.4:
        return "🟢 Rising Opportunity", "high"
    if numeric_score >= 30:
        return "🟡 Competitive Area", "mid"
    return "🔴 Saturated Topic", "low"


def term_relevance_score(term: str, query: str) -> float:
    term_key = normalize_topic_key(term)
    query_concepts = semantic_query_concepts(query)
    score = 0.0

    for concept in query_concepts:
        concept_terms = [normalize_topic_key(concept["name"]), *[normalize_topic_key(t) for t in concept["terms"]]]
        if any(concept_term and concept_term in term_key for concept_term in concept_terms):
            score += concept["weight"]

    return score


def list_focus_terms(top_topics: pd.DataFrame, top_keywords: pd.DataFrame, query: str = "") -> list[str]:
    terms = []
    for df, column in [(top_topics, "topic"), (top_keywords, "keyword")]:
        if column in df.columns:
            terms.extend(df[column].dropna().astype(str).head(5).tolist())
    seen = set()
    clean_terms = []
    for term in terms:
        normalized = normalize_topic_key(term)
        key = normalized
        if key and key != "unknown" and key not in seen:
            seen.add(key)
            clean_terms.append(clean_topic_label(term))

    if query:
        clean_terms = sorted(
            clean_terms,
            key=lambda term: (term_relevance_score(term, query), term.lower()),
            reverse=True,
        )

    return clean_terms[:5]


def normalize_topic_key(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or "").lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("’", "'").replace("`", "'").replace("´", "'")
    text = re.sub(r"\balzheimer\?s\b", "alzheimer", text)
    text = re.sub(r"\balzheimer'?s\b", "alzheimer", text)
    text = re.sub(r"\balzheimer\s+s\b", "alzheimer", text)
    text = re.sub(r"\bdisease\b", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_topic_label(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "").replace("’", "'")).strip(" ,;")
    text = re.sub(r"\bAlzheimer\?s disease\b", "Alzheimer's disease", text, flags=re.I)
    text = re.sub(r"\bAlzheimer's disease\b", "Alzheimer's disease", text, flags=re.I)
    return text


def current_selected_domain() -> str:
    return st.session_state.get("selected_research_domain", HEALTHCARE_DOMAIN)


def contains_domain_term(text: str, terms: set[str]) -> bool:
    key = normalize_topic_key(text)
    normalized_terms = [normalize_topic_key(term) for term in terms]
    return any(term and re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", key) for term in normalized_terms)


def matched_domain_terms(text: str, terms: set[str] | list[str]) -> list[str]:
    key = normalize_topic_key(text)
    matches = []
    for term in terms:
        normalized = normalize_topic_key(term)
        if normalized and re.search(rf"(?<![a-z0-9]){re.escape(normalized)}(?![a-z0-9])", key):
            matches.append(term)
    return sorted(matches)


def is_engineering_health_hybrid(text: str) -> bool:
    return contains_domain_term(text, ENGINEERING_HEALTH_HYBRID_TERMS)


def infer_research_domain(text: str) -> str:
    key = normalize_topic_key(text)
    if contains_domain_term(key, UNSUPPORTED_DOMAIN_TERMS):
        return "Unsupported"
    healthcare_hits = len(matched_domain_terms(key, HEALTHCARE_KEYWORDS))
    engineering_hits = len(matched_domain_terms(key, ENGINEERING_KEYWORDS))
    if healthcare_hits and engineering_hits:
        return "Healthcare-Engineering Hybrid" if is_engineering_health_hybrid(key) else (
            HEALTHCARE_DOMAIN if healthcare_hits >= engineering_hits else ENGINEERING_DOMAIN
        )
    if healthcare_hits:
        return HEALTHCARE_DOMAIN
    if engineering_hits:
        return ENGINEERING_DOMAIN
    return "Not detected"


def validate_domain_query(query: str, selected_domain: str) -> tuple[bool, str, dict]:
    inferred = infer_research_domain(query)
    blocked_terms = matched_domain_terms(query, BLOCKED_KEYWORDS)
    healthcare_terms = matched_domain_terms(query, HEALTHCARE_KEYWORDS)
    engineering_terms = matched_domain_terms(query, ENGINEERING_KEYWORDS)
    selected_terms = engineering_terms if selected_domain == ENGINEERING_DOMAIN else healthcare_terms
    debug = {
        "selected_domain": selected_domain,
        "inferred_domain": inferred,
        "domain_match": bool(selected_terms),
        "leakage_terms": blocked_terms,
        "engineering_terms": engineering_terms,
        "healthcare_terms": healthcare_terms,
        "classification_confidence": "high" if selected_terms else "low",
    }
    if blocked_terms:
        debug["domain_match"] = False
        return False, DOMAIN_SUPPORT_WARNING, debug
    if selected_terms:
        return True, "", debug
    return True, "Topic classification confidence is low, but analysis can still continue.", debug


def forbidden_terms_for_domain(selected_domain: str, query: str) -> set[str]:
    if selected_domain == HEALTHCARE_DOMAIN:
        return HEALTHCARE_BLACKLIST
    if selected_domain == ENGINEERING_DOMAIN and not is_engineering_health_hybrid(query):
        return ENGINEERING_BIOMED_BLACKLIST
    return set()


def domain_specific_strategy(query: str, selected_domain: str) -> dict[str, str]:
    key = normalize_topic_key(query)
    if selected_domain == ENGINEERING_DOMAIN:
        if any(term in key for term in ["uav", "drone", "swarm"]):
            return {
                "direction": "Real-time UAV swarm threat detection using edge AI and computer vision",
                "methodology": "computer vision; edge AI architecture; sensor fusion; anomaly detection; benchmark dataset evaluation",
                "evidence": "UAV video streams; simulated swarm scenarios; edge-device latency benchmarks; detection robustness tests",
                "differentiation": "Differentiate the work through real-time deployment constraints, adversarial scenarios, and benchmarked swarm-level validation.",
            }
        if any(term in key for term in ["wind turbine", "digital twin", "predictive maintenance"]):
            return {
                "direction": "Digital twin-enabled predictive maintenance for wind turbine reliability",
                "methodology": "digital twin modeling; fault diagnosis; time-series forecasting; sensor fusion; reliability analysis",
                "evidence": "SCADA signals; vibration/temperature sensors; turbine fault logs; simulation-based validation",
                "differentiation": "Differentiate the work through physics-informed digital twins, multi-sensor evidence, and season-level reliability validation.",
            }
        return {
            "direction": naturalize_topic_title(query, "engineering validation"),
            "methodology": "predictive maintenance; digital twin modeling; anomaly detection; optimization algorithms; real-time monitoring",
            "evidence": "sensor data; benchmark datasets; simulation-based validation; reliability metrics",
            "differentiation": "Differentiate the work through deployment constraints, robust validation, and measurable engineering performance gains.",
        }
    return build_research_strategy(query, pd.DataFrame())


def domain_specific_insight(query: str, selected_domain: str) -> str:
    if selected_domain == ENGINEERING_DOMAIN:
        return (
            "This engineering topic should be framed around system-level validation, real-time constraints, "
            "sensor or simulation evidence, and reliability-oriented performance metrics. Strong differentiation "
            "comes from benchmark evaluation, deployment feasibility, and robust fault or threat detection."
        )
    return (
        "This healthcare and biomedical topic should be framed around disease-specific evidence, modality-aware modeling, "
        "explainability, and external validation. Strong differentiation comes from clinically meaningful endpoints and "
        "privacy-preserving or multimodal validation when relevant."
    )


def domain_specific_paperability_reason(query: str, selected_domain: str) -> str:
    if selected_domain == ENGINEERING_DOMAIN:
        return "Engineering evidence such as sensor streams, simulation results, benchmark datasets and reliability metrics improves publication feasibility."
    return domain_evidence_reason(query)


def domain_narrowing_for_selected(query: str, selected_domain: str) -> str:
    if selected_domain == ENGINEERING_DOMAIN:
        key = normalize_topic_key(query)
        if "uav" in key or "drone" in key:
            return "Narrow the topic around UAV swarm computer vision, edge AI deployment, real-time threat detection metrics and benchmarked robustness validation."
        if "wind turbine" in key:
            return "Narrow the topic around digital twin modeling, multi-sensor fusion, fault diagnosis and season-level reliability validation for wind turbines."
        return "Narrow the topic around a specific engineering system, measurable performance metric, benchmark dataset and deployment-oriented validation protocol."
    return domain_narrowing_direction(query)


def apply_domain_guard_to_results(results: dict) -> dict:
    selected_domain = results.get("selected_domain") or current_selected_domain()
    query = results.get("query", "")
    forbidden = forbidden_terms_for_domain(selected_domain, query)
    corrected = 0
    leakage_terms: set[str] = set()

    def has_forbidden(text: str) -> bool:
        key = normalize_topic_key(text)
        found = {term for term in forbidden if term in key}
        leakage_terms.update(found)
        return bool(found)

    suggestions = _as_dataframe(results.get("ai_topic_suggestions"))
    if selected_domain == ENGINEERING_DOMAIN:
        results["research_strategy"] = domain_specific_strategy(query, selected_domain)
        corrected += 1

    if not suggestions.empty and forbidden:
        title_col = "suggested_research_topic" if "suggested_research_topic" in suggestions.columns else "suggested_topic" if "suggested_topic" in suggestions.columns else None
        if title_col:
            mask = suggestions.astype(str).agg(" ".join, axis=1).map(lambda text: not has_forbidden(text))
            filtered = suggestions.loc[mask].copy()
            corrected += len(suggestions) - len(filtered)
            if len(filtered) < 3:
                fallback = domain_adapted_suggestions(query) if selected_domain == HEALTHCARE_DOMAIN else pd.DataFrame(
                    [(item["title"], 68, "positive", item["rationale"]) for item in engineering_topic_refinement(query)],
                    columns=["suggested_research_topic", "gap_score", "growth_rate", "recommendation"],
                )
                filtered = pd.concat([filtered, fallback], ignore_index=True).drop_duplicates(subset=[title_col], keep="first")
            results["ai_topic_suggestions"] = filtered.head(8)

    if forbidden and has_forbidden(results.get("ai_research_insight", "")):
        results["ai_research_insight"] = domain_specific_insight(query, selected_domain)
        corrected += 1

    strategy = dict(results.get("research_strategy") or {})
    if forbidden and any(has_forbidden(value) for value in strategy.values()):
        results["research_strategy"] = domain_specific_strategy(query, selected_domain)
        corrected += 1

    paperability = dict(results.get("paperability_score") or {})
    if paperability:
        reasons = [reason for reason in paperability.get("reasons", []) if not has_forbidden(reason)]
        if len(reasons) != len(paperability.get("reasons", [])):
            corrected += len(paperability.get("reasons", [])) - len(reasons)
        domain_reason = domain_specific_paperability_reason(query, selected_domain)
        if selected_domain == ENGINEERING_DOMAIN and domain_reason not in reasons:
            reasons.insert(0, domain_reason)
        if not reasons:
            reasons = [domain_reason]
        paperability["reasons"] = reasons[:5]
        if selected_domain == ENGINEERING_DOMAIN or has_forbidden(paperability.get("recommended_next_action", "")):
            paperability["recommended_next_action"] = domain_narrowing_for_selected(query, selected_domain)
            corrected += 1
        results["paperability_score"] = paperability

    inferred = infer_research_domain(query)
    leakage_score = round(min(1.0, len(leakage_terms) / 5), 2)
    guard = {
        "selected_domain": selected_domain,
        "inferred_domain": inferred,
        "domain_match": inferred in {selected_domain, "Healthcare-Engineering Hybrid", "Not detected"},
        "leakage_terms": sorted(leakage_terms),
        "corrected_items_count": corrected,
        "domain_leakage_risk_score": leakage_score,
    }
    results["domain_guard"] = guard
    domain_reasoning = dict(results.get("domain_reasoning") or {})
    domain_reasoning["selected_domain"] = selected_domain
    domain_reasoning["domain_guard_leakage_score"] = leakage_score
    domain_reasoning["domain_guard_corrected_items"] = corrected
    results["domain_reasoning"] = domain_reasoning
    return results


def _contains_term(text_key: str, term: str) -> bool:
    term_key = normalize_topic_key(term)
    if not term_key:
        return False
    pattern = rf"(?<![a-z0-9]){re.escape(term_key)}(?![a-z0-9])"
    return bool(re.search(pattern, text_key))


def extract_query_concepts(text: str) -> dict[str, list[str]]:
    key = normalize_topic_key(text)
    concepts: dict[str, list[str]] = {}

    for category, entries in CONCEPT_PATTERNS.items():
        values = []
        for label, terms in entries.items():
            if any(_contains_term(key, term) for term in [label, *terms]):
                values.append(label)
        concepts[category] = list(dict.fromkeys(values))

    families = []
    for family, spec in DOMAIN_ONTOLOGY.items():
        if any(_contains_term(key, term) for term in spec["terms"]):
            families.append(family)

    if "breast cancer" in concepts.get("disease", []):
        concepts["disease"] = [item for item in concepts.get("disease", []) if item != "cancer"]
        concepts["clinical_domain"] = list(dict.fromkeys([*concepts.get("clinical_domain", []), "oncology"]))
        if "classification" in concepts.get("task", []):
            concepts["modality"] = list(dict.fromkeys([*concepts.get("modality", []), "medical imaging"]))
            families.append("medical_imaging")
        families.append("oncology")
    if "alzheimer" in concepts.get("disease", []):
        concepts["clinical_domain"] = list(dict.fromkeys([*concepts.get("clinical_domain", []), "neurology"]))
        families.append("neurodegenerative")
    if "depression" in concepts.get("disease", []):
        concepts["clinical_domain"] = list(dict.fromkeys([*concepts.get("clinical_domain", []), "mental health"]))
        families.append("mental_health")
    if "autism" in concepts.get("disease", []):
        concepts["clinical_domain"] = list(dict.fromkeys([*concepts.get("clinical_domain", []), "neurodevelopmental"]))
        families.append("neurodevelopmental")
    if "eeg" in concepts.get("modality", []):
        families.append("signal_processing")
    if "blockchain" in concepts.get("method", []):
        concepts["clinical_domain"] = list(dict.fromkeys([*concepts.get("clinical_domain", []), "healthcare security"]))
        concepts["modality"] = list(dict.fromkeys([*concepts.get("modality", []), "blockchain records"]))
        families.append("healthcare_security")
    if (
        any(term in key for term in ["football", "soccer", "athlete", "player", "injury", "sports", "training load", "workload"])
        or "sports injury" in concepts.get("disease", [])
    ):
        concepts["clinical_domain"] = list(dict.fromkeys([*concepts.get("clinical_domain", []), "sports medicine"]))
        if "sports injury" not in concepts.get("disease", []) and "injury" in key:
            concepts["disease"] = list(dict.fromkeys([*concepts.get("disease", []), "sports injury"]))
        if not concepts.get("modality"):
            concepts["modality"] = ["training load", "match statistics", "wearable sensors"]
        families.append("sports_medicine")

    concepts["families"] = list(dict.fromkeys(families))
    concepts["core_terms"] = domain_core_terms(concepts)
    return concepts


def domain_core_terms(concepts: dict[str, list[str]]) -> list[str]:
    terms = []
    for category in ["disease", "modality", "task", "clinical_domain"]:
        terms.extend(concepts.get(category, []))

    expansions = {
        "breast cancer": ["breast", "cancer", "tumor", "mammography", "pathology", "biopsy"],
        "alzheimer": ["alzheimer", "dementia", "neurodegeneration", "mri", "pet", "cognition"],
        "depression": ["depression", "eeg", "biosignal", "spectral"],
        "autism": ["autism", "asd", "neurodevelopmental", "eeg", "eye tracking", "gaze"],
        "sports injury": ["football", "soccer", "athlete", "injury risk", "training load", "workload", "gps tracking", "wearable sensors"],
        "sports medicine": ["football", "soccer", "athlete monitoring", "injury prediction", "player workload", "performance analytics"],
        "healthcare security": ["blockchain", "privacy", "security", "interoperability"],
    }
    for item in list(terms):
        terms.extend(expansions.get(item, []))
    return list(dict.fromkeys(normalize_topic_key(term) for term in terms if term))


def _concept_overlap(query_values: list[str], candidate_values: list[str], neutral: float = 0.65) -> float:
    if not query_values:
        return neutral
    if not candidate_values:
        return 0.55

    query_set = set(query_values)
    candidate_set = set(candidate_values)
    if query_set & candidate_set:
        return 1.0
    return 0.12


def domain_consistency_score(query: str, candidate_text: str) -> tuple[float, dict]:
    query_concepts = extract_query_concepts(query)
    candidate_concepts = extract_query_concepts(candidate_text)
    candidate_key = normalize_topic_key(candidate_text)

    disease_score = _concept_overlap(query_concepts.get("disease", []), candidate_concepts.get("disease", []))
    modality_score = _concept_overlap(query_concepts.get("modality", []), candidate_concepts.get("modality", []))
    method_score = _concept_overlap(query_concepts.get("method", []), candidate_concepts.get("method", []), neutral=0.7)
    clinical_score = _concept_overlap(query_concepts.get("clinical_domain", []), candidate_concepts.get("clinical_domain", []))
    family_score = _concept_overlap(query_concepts.get("families", []), candidate_concepts.get("families", []))

    core_terms = query_concepts.get("core_terms", [])
    core_hits = sum(1 for term in core_terms if term and term in candidate_key)
    core_score = min(1.0, core_hits / max(1, min(len(core_terms), 4))) if core_terms else 0.7

    score = (
        disease_score * 0.25
        + modality_score * 0.22
        + method_score * 0.15
        + clinical_score * 0.18
        + family_score * 0.10
        + core_score * 0.10
    )

    leakage_hits = [
        term for term in LEAKAGE_TERMS
        if term in candidate_key and term not in normalize_topic_key(query)
    ]
    if leakage_hits and not (set(query_concepts.get("families", [])) & set(candidate_concepts.get("families", []))):
        score -= 0.25

    mismatch = (
        bool(query_concepts.get("disease") and candidate_concepts.get("disease") and disease_score < 0.2)
        or bool(query_concepts.get("modality") and candidate_concepts.get("modality") and modality_score < 0.2)
        or bool(query_concepts.get("clinical_domain") and candidate_concepts.get("clinical_domain") and clinical_score < 0.2)
    )
    if mismatch:
        score -= 0.20

    details = {
        "query_concepts": query_concepts,
        "candidate_concepts": candidate_concepts,
        "leakage_hits": leakage_hits,
        "mismatch": mismatch,
        "core_overlap": round(core_score, 2),
    }
    return round(max(0.0, min(score, 1.0)), 3), details


def domain_label(values: list[str], fallback: str = "Not detected") -> str:
    if not values:
        return fallback
    return ", ".join(title_case_topic(value) for value in values[:3])


def build_domain_reasoning(query: str, suggestions: pd.DataFrame | None = None) -> dict:
    concepts = extract_query_concepts(query)
    suggestions = _as_dataframe(suggestions)
    scores = pd.to_numeric(suggestions.get("domain_consistency_score", pd.Series(dtype=float)), errors="coerce").dropna()
    consistency = float(scores.head(5).mean()) if not scores.empty else 0.75
    leakage_filtered = int(suggestions.attrs.get("domain_filtered_count", 0)) if hasattr(suggestions, "attrs") else 0

    if consistency >= 0.75:
        consistency_label = "High"
        risk = "Low" if leakage_filtered == 0 else "Medium"
    elif consistency >= 0.50:
        consistency_label = "Medium"
        risk = "Medium"
    else:
        consistency_label = "Low"
        risk = "High"

    method = concepts.get("method", [])
    modality = concepts.get("modality", [])
    dominant_method = domain_label(method, "General AI / ML")
    if "cnn" in method and ("medical imaging" in modality or "mri" in modality):
        dominant_method = "CNN-based imaging"
    elif "vision transformer" in method:
        dominant_method = "Transformer-based imaging"
    elif "blockchain" in method:
        dominant_method = "Blockchain-based security"

    return {
        "primary_disease": domain_label(concepts.get("disease", [])),
        "primary_modality": domain_label(modality),
        "primary_method": dominant_method,
        "clinical_domain": domain_label(concepts.get("clinical_domain", [])),
        "domain_consistency_score": round(consistency, 3),
        "domain_consistency": consistency_label,
        "semantic_leakage_risk": risk,
        "leakage_filtered_count": leakage_filtered,
        "concepts": concepts,
    }


def domain_reasoning_to_dataframe(reasoning: dict | None) -> pd.DataFrame:
    reasoning = reasoning or {}
    rows = [
        ("primary disease", reasoning.get("primary_disease", "-")),
        ("modality", reasoning.get("primary_modality", "-")),
        ("clinical domain", reasoning.get("clinical_domain", "-")),
        ("dominant methodology", reasoning.get("primary_method", "-")),
        ("domain consistency", reasoning.get("domain_consistency", "-")),
        ("domain consistency score", reasoning.get("domain_consistency_score", "-")),
        ("semantic leakage risk", reasoning.get("semantic_leakage_risk", "-")),
        ("filtered leakage suggestions", reasoning.get("leakage_filtered_count", 0)),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def trend_is_rising(trend: pd.DataFrame) -> bool:
    if trend.empty or "publication_count" not in trend.columns or len(trend) < 2:
        return False
    counts = pd.to_numeric(trend["publication_count"], errors="coerce").fillna(0)
    split = max(1, len(counts) // 2)
    return counts.tail(split).mean() > counts.head(split).mean()


def compute_strategic_opportunity_score(
    gap: dict,
    query: str,
    distribution: dict | None = None,
    openalex_gap: dict | None = None,
    trend: pd.DataFrame | None = None,
    top_topics: pd.DataFrame | None = None,
    top_keywords: pd.DataFrame | None = None,
) -> float:
    raw_score = parse_numeric(gap.get("gap_score"))
    matched = parse_numeric(gap.get("total_records"))
    growth = parse_numeric(gap.get("growth_rate"))
    distribution = distribution or {}
    query_key = normalize_topic_key(query)
    score = raw_score
    trend = _as_dataframe(trend)
    top_topics = _as_dataframe(top_topics)
    top_keywords = _as_dataframe(top_keywords)

    popular_terms = ["alzheimer", "artificial intelligence", "ai", "machine learning", "diagnosis"]
    is_popular_ai_health = "alzheimer" in query_key and (
        "artificial intelligence" in query_key or " ai " in f" {query_key} " or "machine learning" in query_key
    )

    if matched > 50 and is_popular_ai_health:
        score = min(score, 85)

    openalex_total = 0
    if openalex_gap:
        openalex_total = parse_numeric(openalex_gap.get("total_records"))
    openalex_total = max(openalex_total, distribution.get("OpenAlex", 0))

    if openalex_total > 5000:
        score = min(score, 70)
    elif openalex_total > 1000:
        score = min(score, 80)

    if matched < 50 and growth > 0:
        score = max(score, min(raw_score, 92))

    if matched > 30:
        density_factor = min(matched / max(matched + 200, 1), 0.55)
        score = score * (1 - density_factor * 0.35)

    if trend_is_rising(trend):
        score += 5

    diversity = len(list_focus_terms(top_topics, top_keywords, query))
    if diversity >= 4:
        score -= 4

    return round(max(0, min(score, 100)), 2)


def generate_ai_insight(
    gap: dict,
    top_topics: pd.DataFrame | None = None,
    top_keywords: pd.DataFrame | None = None,
    trend: pd.DataFrame | None = None,
    suggestions: pd.DataFrame | None = None,
    distribution: dict | None = None,
    query: str = "",
) -> str:
    score = parse_numeric(gap.get("gap_score"))
    growth = parse_numeric(gap.get("growth_rate"))
    matched = int(parse_numeric(gap.get("total_records")))
    top_topics = _as_dataframe(top_topics)
    top_keywords = _as_dataframe(top_keywords)
    trend = _as_dataframe(trend)
    suggestions = _as_dataframe(suggestions)
    distribution = distribution or {}
    focus_terms = list_focus_terms(top_topics, top_keywords, query)
    focus_text = ", ".join(focus_terms[:3]) if focus_terms else "ilgili alt konular"
    suggestion_text = " ".join(
        suggestions.astype(str).head(5).agg(" ".join, axis=1).tolist()
    ).lower() if not suggestions.empty else ""
    keyword_text = f"{' '.join(focus_terms).lower()} {suggestion_text}"
    special_terms = [
        term for term in ["multimodal", "federated", "explainable", "small dataset"]
        if term in keyword_text
    ]
    rising = trend_is_rising(trend) or growth > 0.4

    if score > 70 and rising:
        opening = "Bu araştırma alanı yükselen ve fırsat potansiyeli taşıyan bir alan görünümündedir."
    elif matched > 1000 and score < 40:
        opening = "Bu araştırma alanı yoğun rekabet ve doygunluk sinyali vermektedir."
    elif rising:
        opening = "Bu araştırma alanı son yıllarda hızlı büyüyen bir alan olarak öne çıkmaktadır."
    else:
        opening = "Bu araştırma alanı orta düzey fırsat potansiyeli taşımaktadır."

    detail = f"Literatür çoğunlukla {focus_text} ekseninde yoğunlaşmaktadır."

    if special_terms:
        highlighted = ", ".join(term.replace("small dataset", "küçük veri setleriyle güvenilir modelleme") for term in special_terms)
        detail += (
            f" Buna karşılık {highlighted} gibi yaklaşımlar daha sınırlı görünmekte ve "
            "daha net bir farklılaşma alanı oluşturmaktadır."
        )
    elif distribution.get("OpenAlex", 0) > distribution.get("PubMed", 0):
        detail += " OpenAlex tarafındaki geniş yayın hacmi, konunun disiplinler arası görünürlüğünün yüksek olduğunu göstermektedir."

    if matched > 1000 and score < 40:
        recommendation = "Daha özgün sonuç için yöntemi, veri setini veya hedef hastalık grubunu daraltmanız önerilir."
    elif score > 70:
        recommendation = "Bu durum, klinik karar destek sistemleri veya yeni yöntem odaklı çalışmalar için güçlü bir araştırma fırsatı oluşturabilir."
    else:
        recommendation = "Daha net bir Research Gap için araştırma sorusunu belirli bir yöntem, veri tipi veya hasta popülasyonu ile sınırlandırın."

    return f"{opening} {detail} {recommendation}"


def build_research_strategy(query: str, suggestions: pd.DataFrame | None = None) -> dict[str, str]:
    text = f"{query} "
    suggestions = _as_dataframe(suggestions)
    if not suggestions.empty:
        text += " ".join(suggestions.astype(str).head(3).agg(" ".join, axis=1).tolist())
    key = text.lower()
    concepts = extract_query_concepts(query)
    disease = concepts.get("disease", [])
    modality = concepts.get("modality", [])
    method = concepts.get("method", [])

    methods = []
    evidence = []
    differentiators = []

    if "breast cancer" in disease:
        direction = "Domain-consistent CNN-based breast cancer image classification"
        evidence.extend(["mammography / histopathology images", "oncology-focused validation", "external test cohort if possible"])
        methods.extend(["EfficientNet / ResNet / DenseNet comparison", "attention-enhanced lightweight CNN", "Grad-CAM explainability"])
        differentiators.extend(["oncology-specific imaging validation", "explainable visual evidence"])
    elif "autism" in disease:
        direction = "Explainable federated multimodal AI for early autism detection"
        evidence.extend(["EEG recordings", "eye-tracking / gaze features", "neurodevelopmental screening cohorts", "external clinical validation"])
        methods.extend(["EEG-eye tracking fusion", "federated learning", "temporal transformer for biosignals", "SHAP/LIME explainability"])
        differentiators.extend(["multimodal neurodevelopmental evidence", "privacy-preserving validation", "explainable screening outputs"])
    elif "depression" in disease and "eeg" in modality:
        direction = "EEG-based depression detection with spectral-temporal validation"
        evidence.extend(["EEG recordings", "spectral/wavelet feature evidence", "subject-level validation"])
        methods.extend(["Wavelet transform + CNN", "spectral attention", "temporal transformer for EEG"])
        differentiators.extend(["signal-processing consistency", "patient-level validation"])
    elif "sports medicine" in concepts.get("clinical_domain", []) or "sports injury" in disease:
        direction = "Explainable deep learning for football player injury risk assessment"
        evidence.extend(["wearable sensor data", "GPS tracking", "training load metrics", "match statistics", "season-level injury records"])
        methods.extend([
            "time-series deep learning",
            "LSTM / GRU / temporal transformer",
            "workload modeling",
            "explainable risk scoring",
            "survival analysis / risk stratification",
        ])
        differentiators.extend(["external team/season validation", "interpretable injury risk scoring"])
    elif "alzheimer" in key:
        direction = "Explainable multimodal AI for early Alzheimer diagnosis"
        evidence.extend(["MRI / PET / clinical records", "small-sample clinical validation"])
    elif "blockchain" in key:
        direction = "Privacy-preserving healthcare security architecture with blockchain"
        evidence.extend(["healthcare transaction logs", "interoperability and auditability evidence"])
    else:
        direction = naturalize_topic_title(query, "")
        evidence.extend(["public datasets", "domain-specific validation evidence"])

    if "transformer" in key or "vision" in key:
        if "eeg" in modality:
            methods.append("Temporal transformer for biosignal sequences")
        elif "sports medicine" in concepts.get("clinical_domain", []):
            methods.append("Temporal transformer for workload sequences")
        else:
            methods.append("Vision Transformer + Grad-CAM/SHAP")
    if "cnn" in method:
        methods.extend(["EfficientNet / ResNet / DenseNet comparison", "lightweight CNN", "attention CNN"])
    if "multimodal" in key:
        methods.append("Multimodal fusion")
        evidence.append("imaging + clinical feature fusion")
    if "federated" in key:
        methods.append("Federated learning")
        evidence.append("privacy-preserving multi-center validation")
    if "explainable" in key or "xai" in key:
        methods.append("SHAP/LIME/Grad-CAM explainability")
    if "small dataset" in key or "small-sample" in key:
        methods.append("Small dataset augmentation + transfer learning")
    if "clinical" in key or "healthcare" in key:
        methods.append("Clinical decision support validation")
    if "blockchain" in key:
        methods.append("Privacy/security/interoperability analysis")
        differentiators.append("veri gizliliği, güvenlik ve birlikte çalışabilirlik")

    if not methods:
        methods = ["Baseline ML/DL comparison", "Robust validation", "Error analysis"]

    if not evidence:
        evidence = ["public datasets", "small-sample clinical validation", "multi-center validation if possible"]

    if "explainable" in key or "xai" in key:
        differentiators.append("açıklanabilirlik")
    if "federated" in key:
        differentiators.append("veri gizliliği")
    if "multimodal" in key:
        differentiators.append("çoklu veri füzyonu")
    if "clinical" in key or "diagnosis" in key:
        differentiators.append("klinik karar desteği")

    if differentiators:
        diff_text = (
            "Mevcut literatürden ayrışmak için yalnızca AI tanısı değil, "
            + ", ".join(dict.fromkeys(differentiators))
            + " birlikte ele alınmalıdır."
        )
    else:
        diff_text = "Mevcut literatürden ayrışmak için yöntem, veri kaynağı ve doğrulama tasarımı birlikte netleştirilmelidir."

    return {
        "direction": direction,
        "methodology": "; ".join(dict.fromkeys(methods[:5])),
        "evidence": "; ".join(dict.fromkeys(evidence[:5])),
        "differentiation": diff_text,
    }


def paperability_level(score) -> tuple[str, str, str]:
    value = parse_numeric(score)

    if value < 40:
        return "Low Publication Potential", "Düşük yayın potansiyeli", "low"
    if value < 65:
        return "Moderate Publication Potential", "Orta düzey yayın potansiyeli", "mid"
    if value < 80:
        return "Strong Publication Potential", "Güçlü yayın potansiyeli", "high"

    return "Very Strong SCI Potential", "Çok güçlü SCI potansiyeli", "high"


def domain_narrowing_direction(query: str, domain_reasoning: dict | None = None) -> str:
    concepts = extract_query_concepts(query)
    disease = concepts.get("disease", [])
    modality = concepts.get("modality", [])
    method = concepts.get("method", [])

    if "autism" in disease and ("eeg" in modality or "eye tracking" in modality):
        return (
            "Konuyu SCI düzeyinde güçlendirmek için EEG-eye tracking fusion, explainability validation, "
            "federated privacy-preserving learning ve external neurodevelopmental screening validation ile daraltın."
        )
    if "breast cancer" in disease:
        return (
            "Konuyu SCI düzeyinde güçlendirmek için mammography veya histopathology tabanlı veri odağı, "
            "explainable CNN mimarileri ve external oncology validation ile daraltın."
        )
    if "alzheimer" in disease and ("mri" in modality or "pet" in modality or "medical imaging" in modality):
        return (
            "Konuyu SCI düzeyinde güçlendirmek için multimodal MRI/PET fusion, explainability validation "
            "ve neuroimaging-based clinical decision support çıktılarıyla daraltın."
        )
    if "blockchain" in method:
        return (
            "Konuyu SCI düzeyinde güçlendirmek için privacy, interoperability, smart contract validation "
            "ve healthcare data governance bileşenleriyle daraltın."
        )
    if "sports medicine" in concepts.get("clinical_domain", []) or "sports injury" in disease:
        return (
            "Konuyu SCI düzeyinde güçlendirmek için wearable sensor data, GPS tracking, training load metrics, "
            "explainable risk scoring ve external team/season injury validation ile daraltın."
        )
    if "depression" in disease and "eeg" in modality:
        return (
            "Konuyu SCI düzeyinde güçlendirmek için EEG spectral-temporal modeling, subject-level validation "
            "ve explainable biosignal evidence ile daraltın."
        )
    return "Yayın potansiyelini artırmak için araştırma sorusunu belirli bir yöntem, veri seti ve doğrulama protokolüyle sınırlandırın."


def domain_evidence_reason(query: str) -> str:
    concepts = extract_query_concepts(query)
    disease = concepts.get("disease", [])
    modality = concepts.get("modality", [])
    method = concepts.get("method", [])

    if "autism" in disease and ("eeg" in modality or "eye tracking" in modality):
        return "EEG recordings, eye-tracking/gaze features ve neurodevelopmental cohorts kanıt uygulanabilirliğini artırır."
    if "breast cancer" in disease:
        return "Mammography, histopathology images ve external oncology validation yayın uygulanabilirliğini artırır."
    if "alzheimer" in disease and ("mri" in modality or "pet" in modality or "medical imaging" in modality):
        return "MRI/PET imaging cohorts ve neuroimaging-based clinical validation yayın uygulanabilirliğini artırır."
    if "blockchain" in method:
        return "Privacy, interoperability, auditability ve real-world healthcare data governance yayın uygulanabilirliğini artırır."
    if "sports medicine" in concepts.get("clinical_domain", []) or "sports injury" in disease:
        return "Wearable sensor data, GPS tracking, training load metrics and season-level injury records improve evidence feasibility."
    if "depression" in disease and "eeg" in modality:
        return "EEG recordings, spectral features ve subject-level validation yayın uygulanabilirliğini artırır."
    return "Domain-specific datasets ve dış doğrulama tasarımı yayın uygulanabilirliğini artırır."


def build_paperability_score(
    query: str,
    gap: dict,
    strategic_score,
    research_strategy: dict,
    suggestions: pd.DataFrame | None = None,
    openalex_gap: dict | None = None,
    distribution: dict | None = None,
    domain_reasoning: dict | None = None,
) -> dict:
    text_parts = [
        query,
        research_strategy.get("direction", ""),
        research_strategy.get("methodology", ""),
        research_strategy.get("evidence", ""),
        research_strategy.get("differentiation", ""),
    ]
    suggestions = _as_dataframe(suggestions)
    if not suggestions.empty:
        text_parts.append(" ".join(suggestions.astype(str).head(3).agg(" ".join, axis=1).tolist()))

    text = normalize_topic_key(" ".join(text_parts))
    strategic = parse_numeric(strategic_score)
    matched = parse_numeric(gap.get("total_records"))
    openalex_volume = parse_numeric((openalex_gap or {}).get("total_records"))
    distribution = distribution or {}
    domain_reasoning = domain_reasoning or build_domain_reasoning(query, suggestions)
    openalex_volume = max(openalex_volume, parse_numeric(distribution.get("OpenAlex", 0)))

    generic_terms = {"artificial intelligence", "machine learning", "deep learning", "healthcare", "diagnosis"}
    query_terms = [token for token in normalize_topic_key(query).split() if len(token) > 2]
    is_generic = len(query_terms) <= 3 or normalize_topic_key(query) in generic_terms

    novelty = strategic
    if is_generic:
        novelty -= 22
    if matched > 500:
        novelty -= 10
    novelty = max(0, min(novelty, 100))

    saturation_risk = min(70, matched * 0.45) + min(30, openalex_volume / 180)
    competition = max(15, 100 - saturation_risk)

    method_terms = ["explainable", "xai", "federated", "multimodal", "transformer", "vision transformer", "clinical validation", "mri", "pet"]
    clinical_terms = ["alzheimer", "cancer", "diagnosis", "clinical decision support", "medical imaging", "neuroimaging", "patient", "disease"]
    evidence_terms = ["mri", "pet", "eeg", "public dataset", "clinical records", "small dataset", "multi center", "multi-center", "validation"]
    differentiation_terms = ["explainability", "açıklanabilirlik", "privacy", "gizlilik", "clinical decision support", "multimodal fusion", "federated learning", "federated"]

    method_hits = [term for term in method_terms if term in text]
    clinical_hits = [term for term in clinical_terms if term in text]
    evidence_hits = [term for term in evidence_terms if term in text]
    differentiation_hits = [term for term in differentiation_terms if term in text]

    methodology = min(100, 38 + len(set(method_hits)) * 10)
    clinical = min(100, 35 + len(set(clinical_hits)) * 12)
    evidence = min(100, 34 + len(set(evidence_hits)) * 11)
    differentiation = min(100, 36 + len(set(differentiation_hits)) * 12)

    total = (
        novelty * 0.24
        + competition * 0.20
        + methodology * 0.18
        + clinical * 0.16
        + evidence * 0.12
        + differentiation * 0.10
    )

    domain_consistency = parse_numeric(domain_reasoning.get("domain_consistency_score"), 0.75)
    leakage_risk = str(domain_reasoning.get("semantic_leakage_risk", "Low")).lower()
    if domain_consistency < 0.55:
        total -= 16
        methodology = max(0, methodology - 12)
    elif domain_consistency < 0.70:
        total -= 7
    else:
        total += 3

    if leakage_risk == "high":
        total -= 14
    elif leakage_risk == "medium":
        total -= 6

    if matched > 50:
        total = min(total, 82)
    if openalex_volume > 5000:
        total = min(total, 72)
    elif openalex_volume > 1000:
        total = min(total, 78)
    if is_generic:
        total = min(total, 68)

    total = round(max(0, min(total, 100)), 2)
    level_en, level_tr, level_key = paperability_level(total)

    reasons = []
    if strategic >= 65:
        reasons.append("Stratejik fırsat skoru yayın fikrinin özgünleştirilebilir olduğunu gösteriyor.")
    elif strategic >= 40:
        reasons.append("Fırsat sinyali orta düzeyde; konu daha net bir yöntem veya veri odağıyla güçlenebilir.")
    else:
        reasons.append("Novelty sinyali sınırlı; yayın potansiyeli için araştırma sorusu daraltılmalı.")

    if saturation_risk > 55:
        reasons.append("Literatür hacmi yüksek olduğu için rekabet riski toplam skoru dengeliyor.")
    elif matched > 0:
        reasons.append("İlgili literatür mevcut ancak konu hâlâ farklılaştırılabilir görünüyor.")

    if method_hits:
        reasons.append("Güçlü yöntem unsurları mevcut: " + ", ".join(dict.fromkeys(method_hits[:4])) + ".")
    if clinical_hits:
        reasons.append("Klinik/pratik bağlam SCI yayın potansiyelini destekliyor.")
    if evidence_hits:
        reasons.append(domain_evidence_reason(query))
    if domain_consistency < 0.65 or leakage_risk in {"medium", "high"}:
        reasons.append("Domain consistency sinyali düşük olduğu için semantic leakage riski skoru aşağı çekiyor.")
    else:
        reasons.append("Domain consistency güçlü; öneriler hastalık, modalite ve yöntem bağlamında uyumlu kalıyor.")

    reasons = reasons[:5]
    if not reasons:
        reasons = ["Bu değerlendirme, konu kapsamı ve mevcut literatür sinyallerine göre tahmini olarak üretildi."]

    next_action = domain_narrowing_direction(query, domain_reasoning)

    metrics = [
        {"metric": "Novelty Potential", "score": round(novelty, 2), "comment": "Strategic Opportunity Score ve konu genelliğine göre tahmini novelty."},
        {"metric": "Competition / Saturation Risk", "score": round(competition, 2), "comment": "Yüksek skor daha düşük rekabet riski anlamına gelir."},
        {"metric": "Methodological Strength", "score": round(methodology, 2), "comment": "Explainable, federated, multimodal, transformer, MRI/PET gibi yöntem sinyalleri."},
        {"metric": "Clinical / Practical Relevance", "score": round(clinical, 2), "comment": "Hastalık, tanı, klinik karar desteği ve medikal görüntüleme bağlamı."},
        {"metric": "Dataset / Evidence Feasibility", "score": round(evidence, 2), "comment": "MRI/PET/EEG, public dataset, clinical records ve validasyon uygulanabilirliği."},
        {"metric": "Differentiation Strength", "score": round(differentiation, 2), "comment": "Açıklanabilirlik, veri gizliliği, klinik karar desteği ve fusion/federated ayrışması."},
    ]

    return {
        "total_score": total,
        "level": level_en,
        "level_tr": level_tr,
        "level_key": level_key,
        "reasons": reasons,
        "recommended_next_action": next_action,
        "metrics": metrics,
        "note": "Bu skor karar destek amaçlı tahmini bir değerlendirmedir; yayın garantisi anlamına gelmez.",
    }


def paperability_to_dataframe(paperability: dict | None) -> pd.DataFrame:
    paperability = paperability or {}
    rows = [
        {
            "metric": "Paperability Score",
            "score": paperability.get("total_score", 0),
            "comment": paperability.get("level_tr", "-"),
        }
    ]
    rows.extend(paperability.get("metrics", []))
    rows.append({
        "metric": "Recommended next action",
        "score": "",
        "comment": paperability.get("recommended_next_action", "-"),
    })
    return pd.DataFrame(rows, columns=["metric", "score", "comment"])


def render_metric_card(label: str, value, note: str = "", badge: str = "", badge_level: str = "mid") -> None:
    badge_html = f'<span class="rm-status rm-status-{badge_level}">{badge}</span>' if badge else ""
    st.markdown(
        f"""
        <div class="rm-card">
            <div class="rm-card-label">{label}</div>
            <div class="rm-card-value">{value}</div>
            <div class="rm-card-note">{note}</div>
            {badge_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def clean_display_table(df: pd.DataFrame, label: str) -> tuple[pd.DataFrame, str]:
    if df.empty:
        return df, f"{label} için veri bulunamadı."

    first_col = df.columns[0]
    clean = df[
        ~df[first_col].fillna("").astype(str).str.strip().str.lower().isin({"", "unknown", "nan"})
    ].copy()

    if clean.empty:
        if label == "ülke":
            return clean, "Ülke bilgisi bu kaynakta yeterli düzeyde bulunamadı."
        return clean, "Veri bulunamadı."

    return clean, ""


def source_distribution(df: pd.DataFrame) -> dict[str, int]:
    if df.empty:
        return {"PubMed": 0, "OpenAlex": 0, "Yerel": 0}

    if "source" not in df.columns:
        return {"PubMed": 0, "OpenAlex": 0, "Yerel": len(df)}

    values = df["source"].fillna("").astype(str).str.strip().str.lower()
    return {
        "PubMed": int(values.eq("pubmed").sum()),
        "OpenAlex": int(values.eq("openalex").sum()),
        "Yerel": int((values.eq("") | values.eq("unknown") | values.eq("yerel")).sum()),
    }


def preprocess_research_query(query: str) -> str:
    text = re.sub(r"\s+", " ", str(query or "").replace("\n", " ")).strip()
    text = re.sub(r"\s*,\s*", ",", text)
    text = re.sub(r",{2,}", ",", text).strip(" ,")

    if "," in text:
        seen = set()
        parts = []
        for part in text.split(","):
            clean = re.sub(r"\s+", " ", part).strip()
            key = clean.lower()
            if clean and key not in seen:
                seen.add(key)
                parts.append(clean)
        text = " ".join(parts)

    words = []
    seen_words = set()
    for word in text.split():
        key = word.lower()
        if key not in seen_words:
            seen_words.add(key)
            words.append(word)

    return " ".join(words)


def title_case_topic(text: str) -> str:
    small_words = {"and", "or", "for", "in", "with", "of", "the", "a", "an", "to", "based"}
    acronyms = {"ai", "mri", "pet", "ct", "eeg", "nlp", "ehr", "dna", "rna"}
    words = re.split(r"\s+", str(text or "").strip())
    titled = []

    for index, word in enumerate(words):
        lower = word.lower()
        if lower in acronyms:
            titled.append(lower.upper())
        elif index > 0 and lower in small_words:
            titled.append(lower)
        else:
            titled.append(lower.capitalize())

    return " ".join(titled)


def naturalize_topic_title(base_query: str, topic: str = "") -> str:
    query = preprocess_research_query(base_query)
    query_key = normalize_topic_key(query)
    tokens = query.lower().split()
    topic_clean = preprocess_research_query(topic)
    topic_key = normalize_topic_key(topic_clean)
    concepts = extract_query_concepts(query)
    disease = concepts.get("disease", [])
    modality = concepts.get("modality", [])
    method = concepts.get("method", [])

    has_blockchain = "blockchain" in query_key
    has_ai = any(token in tokens for token in ["ai", "artificial", "intelligence"])
    has_explainable = any(token in tokens for token in ["explainable", "xai"])
    has_alzheimer = "alzheimer" in query_key
    has_healthcare = any(token in tokens for token in ["healthcare", "medical", "clinical", "medicine"])
    has_security = "security" in query_key or "security" in topic_clean.lower()

    if has_explainable and has_alzheimer and "transformer" in topic_key:
        return "Transformer-Based Explainable AI for Early Alzheimer Diagnosis"
    if has_explainable and has_alzheimer and "federated" in topic_key:
        return "Federated Explainable AI for Alzheimer Diagnosis"
    if has_explainable and has_alzheimer and "multimodal" in topic_key:
        return "Multimodal Explainable AI for Early Alzheimer Detection"
    if "breast cancer" in disease and "cnn" in method:
        if "efficientnet" in topic_key:
            return "EfficientNet-Based Breast Cancer Image Classification"
        if "resnet" in topic_key:
            return "ResNet-Based Breast Cancer Mammography Classification"
        if "attention" in topic_key:
            return "Attention-Enhanced CNN for Breast Cancer Classification"
        if "histopathology" in topic_key:
            return "CNN-Based Breast Cancer Histopathology Classification"
    if "depression" in disease and "eeg" in modality:
        if "wavelet" in topic_key:
            return "Wavelet-Based EEG Depression Detection"
        if "spectral" in topic_key or "attention" in topic_key:
            return "Spectral Attention Models for EEG-Based Depression Detection"
        return "EEG-Based Depression Detection with Temporal Deep Learning"
    if has_blockchain and topic_clean and (
        "blockchain" in topic_key
        or "smart contract" in topic_key
        or "interoperable" in topic_key
        or "privacy" in topic_key
    ):
        return title_case_topic(topic_clean)
    if has_blockchain and has_ai and has_healthcare and has_security:
        return "AI-based Healthcare Security with Blockchain"
    if has_blockchain and has_ai and has_healthcare:
        return "AI-based Healthcare Intelligence with Blockchain"
    if has_ai and has_alzheimer and topic_clean:
        return f"Explainable AI for Alzheimer {title_case_topic(topic_clean)}"
    if has_ai and topic_clean:
        return f"AI-based {title_case_topic(topic_clean)}"
    if topic_clean:
        return f"{title_case_topic(query)} for {title_case_topic(topic_clean)}"

    return title_case_topic(query)


def domain_adapted_suggestions(query: str) -> pd.DataFrame:
    concepts = extract_query_concepts(query)
    disease = concepts.get("disease", [])
    modality = concepts.get("modality", [])
    method = concepts.get("method", [])
    task = concepts.get("task", [])

    if "breast cancer" in disease:
        rows = [
            ("Attention-Enhanced CNN for Breast Cancer Classification", 74, "positive", "Oncology and imaging domain retained; attention CNN can improve differentiation."),
            ("EfficientNet-Based Breast Cancer Mammography Classification", 71, "positive", "Lightweight CNN family fits breast imaging classification workflows."),
            ("Explainable ResNet for Breast Cancer Histopathology Classification", 69, "positive", "Adds explainability and pathology evidence without leaving oncology."),
        ]
    elif "alzheimer" in disease and ("mri" in modality or "medical imaging" in modality):
        rows = [
            ("Transformer-Based Explainable AI for Early Alzheimer MRI Diagnosis", 76, "positive", "Keeps the focus on Alzheimer MRI, XAI, and neuroimaging validation."),
            ("Multimodal MRI/PET Fusion for Alzheimer Progression Analysis", 73, "positive", "Adds clinically plausible imaging fusion within neurology."),
            ("Grad-CAM Validated Vision Transformers for Alzheimer MRI Analysis", 70, "positive", "Strengthens explainability for imaging-based diagnosis."),
        ]
    elif "depression" in disease and "eeg" in modality:
        rows = [
            ("Spectral Attention Models for EEG-Based Depression Detection", 72, "positive", "Signal-processing family retained through spectral EEG features."),
            ("Wavelet-CNN Framework for EEG Depression Classification", 69, "positive", "Wavelet and CNN methods fit biosignal evidence."),
            ("Temporal Transformer for EEG-Based Depression Screening", 67, "positive", "Transformer use is adapted to sequential EEG signals."),
        ]
    elif "autism" in disease:
        rows = [
            ("Federated Explainable EEG-Eye Tracking Fusion for Early Autism Detection", 76, "positive", "Keeps the focus on neurodevelopmental screening, multimodal signals, and privacy-aware validation."),
            ("Privacy-Aware Multimodal Neurodevelopmental AI for ASD Screening", 73, "positive", "Combines federated learning, clinical relevance, and autism-specific screening."),
            ("Explainable Temporal Transformers for Autism Biosignal and Gaze Fusion", 71, "positive", "Adapts transformer reasoning to EEG and eye-tracking evidence."),
        ]
    elif "sports medicine" in concepts.get("clinical_domain", []) or "sports injury" in disease:
        rows = [
            ("Explainable Deep Learning for Football Player Injury Risk Prediction", 74, "positive", "Sports medicine domain retained through player workload and injury risk modeling."),
            ("Wearable Sensor-Based Injury Risk Assessment in Professional Football", 71, "positive", "Wearable and GPS evidence supports athlete monitoring and risk scoring."),
            ("Temporal Transformer Models for Training Load and Injury Prediction in Soccer", 69, "positive", "Time-series deep learning is adapted to training load and match data."),
        ]
    elif "blockchain" in method:
        rows = [
            ("Privacy-Preserving Blockchain Framework for Healthcare Data Security", 72, "positive", "Focuses on privacy, security, and healthcare interoperability."),
            ("Smart Contract-Based Healthcare Access Control and Auditability", 68, "positive", "Keeps the method within blockchain governance and security."),
            ("Interoperable Blockchain Architecture for Secure Medical Records", 66, "positive", "Targets practical healthcare data exchange evidence."),
        ]
    else:
        title = naturalize_topic_title(query, "clinical validation" if "diagnosis" in task else "robust validation")
        rows = [
            (title, 62, "neutral", "Domain-aware fallback generated from the detected query concepts."),
        ]

    out = pd.DataFrame(rows, columns=["suggested_research_topic", "gap_score", "growth_rate", "recommendation"])
    scores = [domain_consistency_score(query, row["suggested_research_topic"])[0] for _, row in out.iterrows()]
    out["domain_consistency_score"] = scores
    out["leakage_risk"] = ["Low" if score >= 0.75 else "Medium" for score in scores]
    return out


def apply_domain_reasoning_filter(suggestions: pd.DataFrame, query: str) -> pd.DataFrame:
    suggestions = _as_dataframe(suggestions)
    fallback = domain_adapted_suggestions(query)

    if suggestions.empty:
        fallback.attrs["domain_filtered_count"] = 0
        return fallback

    out = suggestions.copy()
    title_col = "suggested_research_topic" if "suggested_research_topic" in out.columns else "suggested_topic" if "suggested_topic" in out.columns else None
    scores = []
    leakage = []

    for _, row in out.iterrows():
        candidate_text = " ".join(str(value) for value in row.fillna("").tolist())
        score, details = domain_consistency_score(query, candidate_text)
        scores.append(score)
        leakage.append("High" if score < 0.45 or details.get("mismatch") else "Medium" if score < 0.65 else "Low")

    out["domain_consistency_score"] = scores
    out["leakage_risk"] = leakage
    before = len(out)
    filtered = out[out["domain_consistency_score"] >= 0.45].copy()

    if title_col:
        filtered = filtered[
            ~filtered[title_col].fillna("").astype(str).map(lambda value: domain_consistency_score(query, value)[1].get("mismatch", False))
        ].copy()

    if len(filtered) < 3:
        filtered = pd.concat([filtered, fallback], ignore_index=True, sort=False)

    if "domain_consistency_score" in filtered.columns:
        filtered = filtered.sort_values("domain_consistency_score", ascending=False)

    if title_col and title_col in filtered.columns:
        filtered = filtered.drop_duplicates(subset=[title_col], keep="first")

    filtered = filtered.head(8).reset_index(drop=True)
    filtered.attrs["domain_filtered_count"] = max(0, before - len(out[out["domain_consistency_score"] >= 0.45]))
    return filtered


def naturalize_suggestions(suggestions: pd.DataFrame, base_query: str) -> pd.DataFrame:
    if suggestions.empty:
        return suggestions

    out = suggestions.copy()
    title_col = None

    if "suggested_research_topic" in out.columns:
        title_col = "suggested_research_topic"
    elif "suggested_topic" in out.columns:
        title_col = "suggested_topic"

    if title_col:
        def _format(row):
            raw_title = str(row.get(title_col, ""))
            base_topic = row.get("base_topic", "")
            suffix = raw_title.replace(base_query, "", 1).strip().strip('"')
            topic = base_topic or suffix
            return naturalize_topic_title(base_query, topic)

        out[title_col] = out.apply(_format, axis=1)
        out = out.drop_duplicates(subset=[title_col], keep="first").reset_index(drop=True)

    return out


QUERY_HELP = """
Doğal dilde kısa bir araştırma konusu girin.

Örnek:
- Explainable AI for Alzheimer Diagnosis
- Blockchain-based Healthcare Security
- Federated Learning in Medical Imaging
"""


def render_query_help() -> None:
    st.caption(
        "Doğal dilde kısa bir araştırma konusu girin. "
        "Örnek: Explainable AI for Alzheimer Diagnosis, "
        "Blockchain-based Healthcare Security, Federated Learning in Medical Imaging"
    )


def read_env_value(key: str, env_path: str | Path = ".env") -> str:
    return get_config_value(key, "", env_path)


def config_bool(key: str, default: bool = False) -> bool:
    return get_config_bool(key, default)


def demo_mode_enabled() -> bool:
    return config_bool("DEMO_MODE", False)


def demo_access_enabled() -> bool:
    return config_bool("DEMO_ACCESS_ENABLED", True)


def admin_emails() -> set[str]:
    raw = read_env_value("ADMIN_EMAILS")
    return {normalize_email(item) for item in raw.split(",") if normalize_email(item)}


def admin_password_hash() -> str:
    return read_env_value("ADMIN_PASSWORD_HASH").strip().lower()


def admin_bypass_enabled() -> bool:
    return get_config_bool("ADMIN_BYPASS", False)


def is_admin_email(email: str) -> bool:
    return normalize_email(email) in admin_emails()


def is_admin() -> bool:
    return bool(st.session_state.get("is_admin", False))


def hash_admin_password(password: str) -> str:
    return hashlib.sha256(str(password or "").encode()).hexdigest()


def log_admin_login_attempt(email: str, status: str) -> None:
    append_demo_csv(
        "admin_login_attempts.csv",
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "email": normalize_email(email),
            "status": status,
        },
        ["timestamp", "email", "status"],
    )


def admin_failed_attempts_today(email: str) -> int:
    ensure_demo_dirs()
    path = DEMO_LOGS_DIR / "admin_login_attempts.csv"
    if not path.exists():
        return 0

    try:
        attempts = pd.read_csv(path)
    except Exception:
        return 0

    if attempts.empty or not {"timestamp", "email", "status"}.issubset(attempts.columns):
        return 0

    today = datetime.now().strftime("%Y-%m-%d")
    emails = attempts["email"].fillna("").astype(str).map(normalize_email)
    statuses = attempts["status"].fillna("").astype(str).str.lower()
    dates = attempts["timestamp"].fillna("").astype(str).str.slice(0, 10)
    return int(((emails == normalize_email(email)) & (dates == today) & statuses.eq("failed")).sum())


def admin_login_blocked(email: str) -> bool:
    return admin_failed_attempts_today(email) >= 5


def normalize_email(email: str) -> str:
    return re.sub(r"\s+", "", str(email or "")).strip().lower()


def ensure_demo_dirs() -> None:
    DEMO_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    DEMO_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def append_demo_csv(filename: str, row: dict, columns: list[str]) -> None:
    ensure_demo_dirs()
    path = DEMO_LOGS_DIR / filename
    clean_row = {column: row.get(column, "") for column in columns}
    pd.DataFrame([clean_row], columns=columns).to_csv(
        path,
        mode="a",
        header=not path.exists(),
        index=False,
        encoding="utf-8-sig",
    )


def register_demo_user(form: dict) -> None:
    email = normalize_email(form.get("email", ""))
    append_demo_csv(
        "demo_users.csv",
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "name": str(form.get("name", "")).strip(),
            "email": email,
            "phone": str(form.get("phone", "")).strip(),
            "university": str(form.get("university", "")).strip(),
            "department": str(form.get("department", "")).strip(),
            "title": str(form.get("title", "")).strip(),
            "research_area": str(form.get("research_area", "")).strip(),
            "consent": bool(form.get("consent", False)),
        },
        ["timestamp", "name", "email", "phone", "university", "department", "title", "research_area", "consent"],
    )
    st.session_state["demo_user_registered"] = True
    st.session_state["demo_user_email"] = email
    st.session_state["demo_user_name"] = str(form.get("name", "")).strip()


def demo_usage_path() -> Path:
    ensure_demo_dirs()
    return DEMO_LOGS_DIR / "demo_usage.csv"


def demo_user_used_today(email: str, date_text: str | None = None) -> bool:
    path = demo_usage_path()
    if not path.exists():
        return False

    date_text = date_text or datetime.now().strftime("%Y-%m-%d")
    try:
        usage = pd.read_csv(path)
    except Exception:
        return False

    if usage.empty or not {"email", "date", "status"}.issubset(usage.columns):
        return False

    emails = usage["email"].fillna("").astype(str).map(normalize_email)
    dates = usage["date"].fillna("").astype(str)
    statuses = usage["status"].fillna("").astype(str).str.lower()
    return bool(((emails == normalize_email(email)) & (dates == date_text) & statuses.isin({"success", "cached"})).any())


def log_demo_usage(email: str, config: dict, status: str, export_path: str = "") -> None:
    append_demo_csv(
        "demo_usage.csv",
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "email": normalize_email(email),
            "research_topic": preprocess_research_query(config.get("query", ""))[:160],
            "source": config.get("data_source_label", config.get("data_source", "")),
            "years_back": config.get("years_back", ""),
            "status": status,
            "export_path": export_path,
        },
        ["timestamp", "date", "email", "research_topic", "source", "years_back", "status", "export_path"],
    )


def demo_cache_key(config: dict) -> str:
    query = preprocess_research_query(config.get("query", ""))[:160].lower()
    raw = json.dumps(
        {
            "query": query,
            "source": config.get("data_source", ""),
            "years_back": int(config.get("years_back", 5)),
            "selected_domain": config.get("selected_domain", current_selected_domain()),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def demo_cache_dir(config: dict) -> Path:
    return DEMO_CACHE_DIR / demo_cache_key(config)


def load_demo_cache(config: dict) -> dict | None:
    cache_dir = demo_cache_dir(config)
    metadata_path = cache_dir / "analysis_metadata.json"
    if not metadata_path.exists():
        return None

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    export_path = metadata.get("export_path", "")
    if not export_path or not Path(export_path).exists():
        return None

    results_pickle = cache_dir / "analysis_results.pkl"
    if results_pickle.exists():
        try:
            cached_results = pd.read_pickle(results_pickle)
            if isinstance(cached_results, dict):
                cached_results = cached_results.copy()
                cached_results["warnings"] = [
                    "Bu konu için daha önce oluşturulmuş demo sonucu gösteriliyor.",
                    *cached_results.get("warnings", []),
                ]
                cached_results["cached_demo_result"] = True
                return cached_results
        except Exception:
            pass

    normalized_path = cache_dir / "normalized_dataset.csv"
    df = pd.read_csv(normalized_path) if normalized_path.exists() else pd.DataFrame()
    return {
        "data_source": config.get("data_source", "-"),
        "data_source_label": config.get("data_source_label", config.get("data_source", "-")),
        "selected_domain": config.get("selected_domain", current_selected_domain()),
        "query": preprocess_research_query(config.get("query", "")),
        "raw_query": config.get("query", ""),
        "analysis_time": metadata.get("analysis_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        "normalized_dataset": df,
        "warnings": ["Bu konu için daha önce oluşturulmuş demo sonucu gösteriliyor."],
        "errors": [],
        "diagnostics": {},
        "export_path": export_path,
        "source_distribution": source_distribution(df),
        "cached_demo_result": True,
    }


def save_demo_cache(config: dict, results: dict) -> None:
    export_path = results.get("export_path", "")
    if not export_path:
        return

    cache_dir = demo_cache_dir(config)
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        pd.to_pickle(results, cache_dir / "analysis_results.pkl")
    except Exception:
        pass
    df = _as_dataframe(results.get("normalized_dataset"))
    df.to_csv(cache_dir / "normalized_dataset.csv", index=False, encoding="utf-8-sig")

    export_dir = Path(export_path)
    for filename in ["summary_report.txt"]:
        source = export_dir / filename
        if source.exists():
            shutil.copy2(source, cache_dir / filename)

    executive_matches = sorted(export_dir.glob("ResearchMind_AI_Executive_Report_*.pdf"))
    if executive_matches:
        shutil.copy2(executive_matches[-1], cache_dir / executive_matches[-1].name)

    (cache_dir / "analysis_metadata.json").write_text(
        json.dumps(
            {
                "query": preprocess_research_query(config.get("query", ""))[:160],
                "source": config.get("data_source", ""),
                "years_back": int(config.get("years_back", 5)),
                "selected_domain": config.get("selected_domain", current_selected_domain()),
                "analysis_time": results.get("analysis_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                "export_path": str(Path(export_path).resolve()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def render_demo_registration_gate() -> bool:
    if not demo_mode_enabled():
        return True

    if not demo_access_enabled():
        render_hero(None)
        st.info("ResearchMind AI demo erişimi proje pazarı etkinliği sonrasında kapatılmıştır. İlginiz için teşekkür ederiz.")
        return False

    if st.session_state.get("demo_user_registered"):
        name = st.session_state.get("demo_user_name", "")
        if name:
            st.sidebar.success(f"Hoş geldiniz, {name}")
        if is_admin():
            st.sidebar.info("Admin modu aktif")
        return True

    render_hero(None)
    st.markdown(
        "ResearchMind AI, araştırma fikrinizi canlı literatür verileriyle analiz eder, "
        "Research Gap Score, Paperability Score ve stratejik araştırma önerileri üretir."
    )
    st.subheader("Demo Erişim Formu")
    email = st.text_input("E-posta *", key="demo_email")
    normalized_email = normalize_email(email)
    email_is_admin = is_admin_email(normalized_email)
    admin_bypass = email_is_admin and admin_bypass_enabled()

    with st.form("demo_registration_form"):
        name = st.text_input("Ad Soyad *", key="demo_name")
        university = st.text_input("Üniversite / Kurum *", key="demo_university")
        department = st.text_input("Bölüm / Birim *", key="demo_department")
        title = st.text_input("Unvan / Title *", key="demo_title")
        phone = st.text_input("Telefon", key="demo_phone")
        research_area = st.text_input("Araştırma alanı", key="demo_research_area")
        admin_password = ""
        if email_is_admin and not admin_bypass:
            admin_password = st.text_input("Admin şifresi", type="password", key="demo_admin_password")
        consent = st.checkbox(
            "Verilerimin demo erişimi ve proje iletişimi amacıyla kaydedilmesini kabul ediyorum.",
            key="demo_consent",
        )
        submitted = st.form_submit_button("Demo Erişimi Başlat", type="primary")

    if submitted:
        missing = [
            label for label, value in [
                ("Ad Soyad", name),
                ("E-posta", email),
                ("Üniversite / Kurum", university),
                ("Bölüm / Birim", department),
                ("Unvan / Title", title),
            ]
            if not str(value).strip()
        ]
        if missing:
            st.error("Lütfen zorunlu alanları doldurun: " + ", ".join(missing))
        elif "@" not in normalized_email or "." not in normalized_email:
            st.error("Lütfen geçerli bir e-posta adresi girin.")
        elif not consent:
            st.error("Demo erişimi için veri kullanım onayını işaretlemeniz gerekir.")
        elif email_is_admin and not admin_bypass and admin_login_blocked(normalized_email):
            st.error("Çok fazla hatalı admin şifresi denemesi yapıldı. Lütfen daha sonra tekrar deneyin.")
        elif email_is_admin and not admin_bypass and not admin_password_hash():
            log_admin_login_attempt(normalized_email, "failed")
            st.error("Admin şifre hash ayarı bulunamadı.")
        elif email_is_admin and not admin_bypass and hash_admin_password(admin_password) != admin_password_hash():
            log_admin_login_attempt(normalized_email, "failed")
            st.error("Admin şifresi hatalı.")
        else:
            if email_is_admin:
                st.session_state["is_admin"] = True
                if not admin_bypass:
                    log_admin_login_attempt(normalized_email, "success")
            else:
                st.session_state["is_admin"] = False
            register_demo_user({
                "name": name,
                "email": normalized_email,
                "phone": phone,
                "university": university,
                "department": department,
                "title": title,
                "research_area": research_area,
                "consent": consent,
            })
            st.success("Demo erişimi başlatıldı.")
            st.rerun()

    return False


def normalize_topic_seed(text: str) -> str:
    clean = preprocess_research_query(text)
    replacements = {
        "otizm": "autism",
        "asd": "autism",
        "göz takibi": "eye tracking",
        "goz takibi": "eye tracking",
        "yapay zeka": "artificial intelligence",
        "yapay zekâ": "artificial intelligence",
        "blokzincir": "blockchain",
        "blok zincir": "blockchain",
        "sağlık verisi": "healthcare data",
        "saglik verisi": "healthcare data",
        "meme kanseri": "breast cancer",
        "erken tanı": "early diagnosis",
        "erken tani": "early diagnosis",
        "görüntüleme": "medical imaging",
        "goruntuleme": "medical imaging",
        "kamu harcamalari": "public expenditure",
        "vergi uyumu": "tax compliance",
        "mali surdurulebilirlik": "fiscal sustainability",
        "butce": "budget",
    }
    lowered = clean.lower()
    for source, target in replacements.items():
        lowered = lowered.replace(source, target)
    return title_case_topic(lowered)


def topic_refinement_prompt(seed: str, selected_domain: str = "") -> str:
    domain_text = selected_domain or current_selected_domain()
    return (
        "Convert the user's Turkish, English, or messy keywords into 5 concise Q1/SCI-style English research topics. "
        "Avoid generic titles and avoid broad 'AI-based ...' phrasing. "
        "Each title must include a clear domain, method, data/modality, task, and novelty angle when possible. "
        "Prefer specific publishable angles such as explainability, external validation, privacy-preserving learning, "
        "multimodal fusion, risk stratification, or decision-support validation when relevant. "
        f"Selected research domain: {domain_text}. Stay strictly within this domain. "
        "Write titles that sound like real high-quality journal article titles. "
        "Return only valid JSON in this exact shape: "
        '[{"title":"...","rationale":"..."},{"title":"...","rationale":"..."}]. '
        f"User input: {seed}"
    )


def parse_topic_json(text: str) -> list[dict[str, str]]:
    raw = str(text or "").strip()
    match = re.search(r"\[[\s\S]*\]", raw)
    if match:
        raw = match.group(0)

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []

    topics = []
    for item in payload if isinstance(payload, list) else []:
        if not isinstance(item, dict):
            continue
        title = re.sub(r"\s+", " ", str(item.get("title", ""))).strip()
        rationale = re.sub(r"\s+", " ", str(item.get("rationale", ""))).strip()
        if title and not is_generic_paper_title(title, title):
            topics.append({"title": title, "rationale": rationale or "Domain-aware SCI-style topic suggestion."})
        if len(topics) >= 5:
            break
    return topics


def engineering_topic_refinement(seed: str) -> list[dict[str, str]]:
    key = normalize_topic_key(normalize_topic_seed(seed))
    if any(term in key for term in ["uav", "drone", "swarm", "robotics"]):
        titles = [
            "Edge AI-Based Computer Vision for Real-Time UAV Swarm Threat Detection",
            "Vision Transformer Models for Autonomous UAV Swarm Surveillance and Threat Assessment",
            "Multi-Sensor Fusion for Robust Drone Swarm Detection in Edge Computing Systems",
            "Real-Time Anomaly Detection for UAV Swarm Security Using Deep Learning",
            "Benchmark Evaluation of Computer Vision Models for UAV Threat Detection",
        ]
    elif any(term in key for term in ["wind turbine", "digital twin", "predictive maintenance", "fault"]):
        titles = [
            "Digital Twin-Driven Predictive Maintenance for Wind Turbine Fault Diagnosis",
            "Sensor Fusion and Time-Series Forecasting for Wind Turbine Health Monitoring",
            "Physics-Informed Digital Twins for Reliability Analysis in Wind Energy Systems",
            "Anomaly Detection Models for Real-Time Wind Turbine Predictive Maintenance",
            "Benchmark Evaluation of Deep Learning Models for Wind Turbine Fault Prediction",
        ]
    else:
        base = title_case_topic(normalize_topic_seed(seed) or "Engineering Systems")
        titles = [
            f"Digital Twin and Predictive Analytics Framework for {base}",
            f"Edge AI-Based Real-Time Monitoring for {base}",
            f"Sensor Fusion and Anomaly Detection for {base}",
            f"Optimization-Driven Control Strategy for {base}",
            f"Benchmark Dataset Evaluation for Reliable {base}",
        ]
    return [{"title": title, "rationale": "Engineering-focused Q1/SCI-style topic generated within the selected domain."} for title in titles]


def rule_based_topic_refinement(seed: str, selected_domain: str | None = None) -> list[dict[str, str]]:
    selected_domain = selected_domain or current_selected_domain()
    normalized = normalize_topic_seed(seed)
    normalized_key = normalize_topic_key(normalized)

    if selected_domain == ENGINEERING_DOMAIN:
        return engineering_topic_refinement(seed)

    if any(term in normalized_key for term in ["tax compliance", "public expenditure", "fiscal sustainability", "public finance", "budget"]):
        return [
            {
                "title": "AI-Assisted Tax Compliance and Public Expenditure Sustainability Analysis",
                "rationale": "Connects tax compliance, public spending, and fiscal sustainability in a publishable analytics topic.",
            },
            {
                "title": "Machine Learning-Based Fiscal Risk Assessment for Public Finance Sustainability",
                "rationale": "Frames the topic around fiscal risk, predictive modeling, and public finance decision support.",
            },
            {
                "title": "Data-Driven Analysis of Tax Compliance and Government Spending Efficiency",
                "rationale": "Focuses on measurable public finance efficiency and compliance behavior.",
            },
            {
                "title": "Predictive Analytics for Fiscal Sustainability and Public Budget Management",
                "rationale": "Turns the keywords into a clear forecasting and budget management research direction.",
            },
            {
                "title": "Explainable AI for Tax Compliance Risk Detection in Public Finance",
                "rationale": "Adds explainability and risk detection for a stronger SCI-style contribution angle.",
            },
        ]

    suggestions = domain_adapted_suggestions(normalized)
    suggestions = naturalize_suggestions(suggestions, normalized)
    titles = synthesize_paper_titles({"query": normalized, "ai_topic_suggestions": suggestions}, min_count=5)
    topics = [
        {
            "title": title,
            "rationale": "Domain, method, modality, task and novelty angle were inferred from the provided keywords.",
        }
        for title in titles[:5]
    ]
    if len(topics) >= 3:
        return topics

    fallback_base = title_case_topic(normalized or "Applied AI Research")
    fallback_titles = [
        f"Explainable AI Framework for {fallback_base}",
        f"Data-Driven Decision Support for {fallback_base}",
        f"Predictive Analytics and Validation Strategy for {fallback_base}",
    ]
    seen = {normalize_topic_key(item["title"]) for item in topics}
    for title in fallback_titles:
        key = normalize_topic_key(title)
        if key not in seen:
            seen.add(key)
            topics.append({
                "title": title,
                "rationale": "Deterministic local fallback generated a complete research topic.",
            })
        if len(topics) >= 5:
            break
    return topics


def mask_secret(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "missing"
    if len(text) <= 8:
        return "*" * len(text)
    return f"{text[:4]}...{text[-4:]}"


def call_llm_topic_refiner(
    provider: str,
    seed: str,
    api_key: str,
    selected_domain: str = "",
    debug: dict | None = None,
) -> list[dict[str, str]]:
    debug = debug if debug is not None else {}
    debug.update({
        "active_provider": provider,
        "secret_detected": bool(str(api_key or "").strip()),
        "secret_masked": mask_secret(api_key),
        "llm_status": "not_started",
        "fallback_reason": "",
    })

    if not api_key.strip() or provider == "Rule-based":
        debug["llm_status"] = "skipped"
        debug["fallback_reason"] = "No LLM provider selected or API key was not detected."
        return []

    prompt = topic_refinement_prompt(seed, selected_domain)
    headers = {"Content-Type": "application/json"}
    started_at = time.perf_counter()

    def finish_from_response(response: requests.Response, content: str) -> list[dict[str, str]]:
        debug["http_status"] = response.status_code
        debug["response_chars"] = len(content or "")
        debug["raw_response_preview"] = str(content or "")[:700]
        topics = parse_topic_json(content)
        debug["parsed_topic_count"] = len(topics)
        debug["elapsed_ms"] = int((time.perf_counter() - started_at) * 1000)
        if topics:
            debug["llm_status"] = "success"
            debug["fallback_reason"] = ""
        else:
            debug["llm_status"] = "empty_or_unparseable_response"
            debug["fallback_reason"] = "LLM returned content, but no valid topic JSON could be parsed."
        return topics

    def ensure_success(response: requests.Response) -> None:
        if response.status_code >= 400:
            debug["http_status"] = response.status_code
            debug["response_chars"] = len(response.text or "")
            debug["raw_response_preview"] = str(response.text or "")[:700]
            debug["elapsed_ms"] = int((time.perf_counter() - started_at) * 1000)
            debug["llm_status"] = "http_error"
            debug["fallback_reason"] = f"LLM provider returned HTTP {response.status_code}."
        response.raise_for_status()

    if provider == "OpenAI":
        headers["Authorization"] = f"Bearer {api_key}"
        debug["client_status"] = "OpenAI-compatible HTTP client ready"
        debug["model"] = "gpt-4o-mini"
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.35,
                "max_tokens": 700,
            },
            timeout=30,
        )
        ensure_success(response)
        return finish_from_response(response, response.json()["choices"][0]["message"]["content"])

    if provider == "Groq":
        headers["Authorization"] = f"Bearer {api_key}"
        debug["client_status"] = "Groq OpenAI-compatible HTTP client ready"
        debug["model"] = "llama-3.1-8b-instant"
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.35,
                "max_tokens": 700,
            },
            timeout=30,
        )
        ensure_success(response)
        return finish_from_response(response, response.json()["choices"][0]["message"]["content"])

    if provider == "Gemini":
        debug["client_status"] = "Gemini HTTP client ready"
        debug["model"] = "gemini-1.5-flash"
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}",
            headers=headers,
            json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.35}},
            timeout=30,
        )
        ensure_success(response)
        candidates = response.json().get("candidates", [])
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        return finish_from_response(response, " ".join(part.get("text", "") for part in parts))

    debug["llm_status"] = "skipped"
    debug["fallback_reason"] = f"Unsupported provider: {provider}"
    return []


TOPIC_PROVIDER_ENV_KEYS = {
    "OpenAI": "OPENAI_API_KEY",
    "Groq": "GROQ_API_KEY",
    "Gemini": "GEMINI_API_KEY",
}


def auto_topic_provider_from_config() -> tuple[str, str]:
    for provider in ["Groq", "OpenAI", "Gemini"]:
        api_key = read_env_value(TOPIC_PROVIDER_ENV_KEYS[provider])
        if api_key:
            return provider, api_key
    return "Rule-based", ""


def refine_research_topics_legacy(seed: str, provider: str, api_key: str) -> tuple[list[dict[str, str]], str]:
    if not str(seed or "").strip():
        return [], "No input"

    try:
        llm_topics = call_llm_topic_refiner(provider, seed, api_key)
        if llm_topics:
            return llm_topics[:5], f"{provider} LLM refinement"
    except Exception as exc:
        st.session_state["topic_suggester_warning"] = f"LLM önerisi başarısız oldu; rule-based öneriler gösteriliyor. ({exc})"

    if provider != "Rule-based" and api_key:
        st.session_state["topic_suggester_warning"] = (
            "Konu önerme servisi şu anda yanıt vermedi. Yerel öneri motoru ile devam ediliyor."
        )

    fallback_topics = rule_based_topic_refinement(seed)
    return fallback_topics[:5], "Rule-based fallback"


def refine_research_topics(seed: str, provider: str, api_key: str, selected_domain: str | None = None) -> tuple[list[dict[str, str]], str]:
    selected_domain = selected_domain or current_selected_domain()
    debug = {
        "active_provider": provider,
        "selected_domain": selected_domain,
        "inferred_domain": infer_research_domain(seed),
        "secret_detected": bool(str(api_key or "").strip()),
        "secret_masked": mask_secret(api_key),
        "llm_status": "not_started",
        "fallback_reason": "",
        "traceback": "",
    }
    st.session_state["topic_llm_debug"] = debug

    if not str(seed or "").strip():
        debug["llm_status"] = "skipped"
        debug["fallback_reason"] = "No topic seed was entered."
        return [], "No input"

    try:
        llm_topics = call_llm_topic_refiner(provider, seed, api_key, selected_domain, debug)
        if llm_topics:
            fallback_topics = rule_based_topic_refinement(seed, selected_domain)[:5]
            debug["fallback_comparison_count"] = len(fallback_topics)
            debug["fallback_comparison_titles"] = [item.get("title", "") for item in fallback_topics]
            debug["llm_quality_note"] = (
                "LLM output is preferred for more creative Q1/SCI-style synthesis; "
                "fallback remains deterministic and domain-safe."
            )
            return llm_topics[:5], f"{provider} LLM"
    except Exception as exc:
        debug["llm_status"] = "exception"
        debug["fallback_reason"] = debug.get("fallback_reason") or str(exc)
        debug["traceback"] = traceback.format_exc()
        st.session_state["topic_suggester_warning"] = (
            "Konu önerme servisi şu anda yanıt vermedi. Yerel öneri motoru ile devam ediliyor."
        )

    if provider != "Rule-based" and api_key:
        st.session_state["topic_suggester_warning"] = (
            "Konu önerme servisi şu anda yanıt vermedi. Yerel öneri motoru ile devam ediliyor."
        )
        if not debug.get("fallback_reason"):
            debug["fallback_reason"] = "LLM returned no valid topic suggestions."

    fallback_topics = rule_based_topic_refinement(seed, selected_domain)
    debug["fallback_topic_count"] = len(fallback_topics)
    debug["fallback_titles"] = [item.get("title", "") for item in fallback_topics[:5]]
    if provider == "Rule-based":
        debug["fallback_reason"] = debug.get("fallback_reason") or "No provider secret was detected, so local fallback was used."
    return fallback_topics[:5], "Rule-based fallback"


QUERY_WIDGET_KEYS = {
    "Local CSV": "local_analysis_query",
    "OpenAlex Live": "openalex_query",
    "PubMed Live": "pubmed_query",
    "Hybrid: OpenAlex + PubMed": "hybrid_query",
}


def apply_suggested_research_topic(title: str) -> None:
    clean_title = re.sub(r"\s+", " ", str(title or "")).strip()
    if not clean_title:
        return

    st.session_state["pending_research_topic"] = clean_title
    st.session_state["selected_research_topic"] = clean_title
    st.session_state["topic_transfer_notice"] = "Önerilen konu araştırma konusu alanına aktarıldı."


def sync_pending_research_topic(data_source: str) -> None:
    pending = st.session_state.pop("pending_research_topic", "")
    if not pending:
        return

    for key in QUERY_WIDGET_KEYS.values():
        st.session_state[key] = pending

    active_key = QUERY_WIDGET_KEYS.get(data_source)
    if active_key:
        st.session_state[active_key] = pending


def render_topic_suggester() -> None:
    with st.expander("Araştırma Konusu Öner", expanded=False):
        if "topic_suggester_seed" not in st.session_state:
            st.session_state["topic_suggester_seed"] = ""
        seed = st.text_area(
            "Türkçe veya İngilizce anahtar kelimelerinizi yazın",
            placeholder="otizm eeg göz takibi\nalzheimer mri yapay zeka\nblokzincir sağlık verisi",
            key="topic_suggester_seed",
            height=90,
        )
        selected_domain = current_selected_domain()
        if is_admin():
            with st.expander("Konu önerme config durumu", expanded=False):
                for name, env_key in TOPIC_PROVIDER_ENV_KEYS.items():
                    secret_value = read_env_value(env_key)
                    st.caption(f"{name}: {mask_secret(secret_value)}")
        if demo_mode_enabled() and not is_admin():
            provider, api_key = auto_topic_provider_from_config()
        else:
            provider_options = ["Rule-based", "OpenAI", "Groq", "Gemini"]
            default_provider, _ = auto_topic_provider_from_config()
            provider = st.selectbox(
                "LLM sağlayıcı",
                provider_options,
                index=provider_options.index(default_provider) if default_provider in provider_options else 0,
                help="API key yoksa ücretsiz/rule-based mod kullanılır.",
                key="topic_suggester_provider",
            )
        if not (demo_mode_enabled() and not is_admin()):
            api_key = ""
        if provider != "Rule-based" and ((not demo_mode_enabled()) or is_admin()):
            env_key = TOPIC_PROVIDER_ENV_KEYS[provider]
            api_key = st.text_input(
                f"{provider} API key",
                value="",
                type="password",
                help=f"Opsiyonel. Boş bırakılırsa .env içindeki {env_key} kullanılır; yoksa rule-based öneriler çalışır.",
                key=f"topic_suggester_{provider.lower()}_api_key",
            ).strip() or read_env_value(env_key)

        if st.button("Konu Öner", use_container_width=True, key="suggest_research_topics_button"):
            domain_ok, domain_message, domain_debug = validate_domain_query(seed, selected_domain)
            st.session_state["domain_guard_debug"] = domain_debug
            if not domain_ok:
                st.warning(domain_message)
                st.session_state["topic_suggestions"] = []
                st.session_state["topic_suggester_results"] = []
                st.session_state["topic_suggester_mode"] = "Domain blocked"
            else:
                if domain_message:
                    st.warning(domain_message)
                with st.spinner("Konu önerileri üretiliyor..."):
                    topics, mode = refine_research_topics(seed, provider, api_key, selected_domain)
                st.session_state["topic_suggestions"] = topics
                st.session_state["topic_suggester_results"] = topics
                st.session_state["topic_suggester_mode"] = mode

        warning = st.session_state.pop("topic_suggester_warning", "")
        if warning:
            st.warning(warning)

        llm_debug = st.session_state.get("topic_llm_debug", {})
        if llm_debug and is_admin():
            with st.expander("LLM konu önerme debug", expanded=False):
                st.write({
                    "active_provider": llm_debug.get("active_provider"),
                    "secret_detected": llm_debug.get("secret_detected"),
                    "secret_masked": llm_debug.get("secret_masked"),
                    "client_status": llm_debug.get("client_status"),
                    "model": llm_debug.get("model"),
                    "llm_status": llm_debug.get("llm_status"),
                    "http_status": llm_debug.get("http_status"),
                    "response_chars": llm_debug.get("response_chars"),
                    "parsed_topic_count": llm_debug.get("parsed_topic_count"),
                    "elapsed_ms": llm_debug.get("elapsed_ms"),
                    "fallback_reason": llm_debug.get("fallback_reason"),
                })
                if llm_debug.get("fallback_comparison_titles"):
                    st.caption("Fallback comparison titles")
                    st.write(llm_debug.get("fallback_comparison_titles"))
                if llm_debug.get("fallback_titles"):
                    st.caption("Fallback titles used")
                    st.write(llm_debug.get("fallback_titles"))
                if llm_debug.get("raw_response_preview"):
                    st.caption("LLM raw response preview")
                    st.code(llm_debug.get("raw_response_preview"))
                if llm_debug.get("traceback"):
                    st.caption("Traceback")
                    st.code(llm_debug.get("traceback"))

        domain_debug = st.session_state.get("domain_guard_debug", {})
        if domain_debug and is_admin():
            with st.expander("DomainGuard debug", expanded=False):
                st.write({
                    "selected_domain": domain_debug.get("selected_domain"),
                    "inferred_domain": domain_debug.get("inferred_domain"),
                    "domain_match": domain_debug.get("domain_match"),
                    "leakage_terms": domain_debug.get("leakage_terms"),
                    "corrected_items_count": domain_debug.get("corrected_items_count", 0),
                    "topic_suggestion_provider": st.session_state.get("topic_suggester_mode", "-"),
                    "fallback_reason": (st.session_state.get("topic_llm_debug", {}) or {}).get("fallback_reason", "-"),
                })

        topics = st.session_state.get("topic_suggestions", st.session_state.get("topic_suggester_results", []))
        if topics:
            st.caption(f"Öneri modu: {st.session_state.get('topic_suggester_mode', 'Rule-based fallback')}")
            for index, item in enumerate(topics, start=1):
                title = item.get("title", "")
                st.markdown(f"**{index}. {title}**")
                if item.get("rationale"):
                    st.caption(item["rationale"])
                st.button(
                    "Bu konuyu analiz et",
                    key=f"use_suggested_topic_{index}_{hashlib.sha256(title.encode('utf-8')).hexdigest()[:8]}",
                    use_container_width=True,
                    on_click=apply_suggested_research_topic,
                    args=(title,),
                )


def render_config_warnings() -> None:
    missing = []
    if not get_config_value("OPENALEX_EMAIL"):
        missing.append("OPENALEX_EMAIL")
    if not get_config_value("NCBI_EMAIL"):
        missing.append("NCBI_EMAIL")

    if demo_mode_enabled() and is_admin_email(st.session_state.get("demo_user_email", "")) and not admin_bypass_enabled():
        if not get_config_value("ADMIN_PASSWORD_HASH"):
            missing.append("ADMIN_PASSWORD_HASH")

    if missing and ((not demo_mode_enabled()) or is_admin()):
        st.warning(
            "Eksik yapılandırma bulundu: "
            + ", ".join(dict.fromkeys(missing))
            + ". Local geliştirmede .env, Streamlit Cloud'da Secrets alanını kullanın."
        )


def render_admin_demo_management() -> None:
    if not (demo_mode_enabled() and is_admin()):
        return

    ensure_demo_dirs()
    with st.expander("Admin Demo Yönetimi", expanded=False):
        users_path = DEMO_LOGS_DIR / "demo_users.csv"
        usage_path = DEMO_LOGS_DIR / "demo_usage.csv"
        attempts_path = DEMO_LOGS_DIR / "admin_login_attempts.csv"
        user_count = len(pd.read_csv(users_path)) if users_path.exists() else 0
        usage_count = len(pd.read_csv(usage_path)) if usage_path.exists() else 0
        attempt_count = len(pd.read_csv(attempts_path)) if attempts_path.exists() else 0

        st.caption(f"Kayıtlı demo kullanıcıları: {user_count}")
        st.caption(f"Demo analiz kayıtları: {usage_count}")
        st.caption(f"Admin giriş denemeleri: {attempt_count}")
        st.caption(f"Cache klasörü: {DEMO_CACHE_DIR.resolve()}")
        st.caption(f"Log klasörü: {DEMO_LOGS_DIR.resolve()}")

        if st.button("Demo cache temizle", key="admin_clear_demo_cache", use_container_width=True):
            if DEMO_CACHE_DIR.exists():
                shutil.rmtree(DEMO_CACHE_DIR)
            DEMO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            st.success("Demo cache temizlendi.")


@st.cache_data(show_spinner=False)
def load_local_csv(path: str, nrows: int) -> pd.DataFrame:
    return read_csv_light(path, nrows=None if nrows == 0 else int(nrows))


def load_pubmed_live(query: str, max_results: int, years_back: int) -> pd.DataFrame:
    return load_pubmed_live_with_credentials(
        query=query,
        max_results=int(max_results),
        years_back=int(years_back),
        email="",
        api_key="",
    )


def load_pubmed_live_with_credentials(
    query: str,
    max_results: int,
    years_back: int,
    email: str = "",
    api_key: str = "",
) -> pd.DataFrame:
    env_config = get_pubmed_config()
    config = PubMedConfig(
        api_key=str(api_key).strip() or env_config.api_key,
        email=str(email).strip() or env_config.email,
    )
    client = PubMedClient(config=config)
    records = client.search(
        query=query,
        max_results=int(max_results),
        years_back=int(years_back),
    )
    return normalize_pubmed_to_researchmind_schema(pd.DataFrame(records))


def normalize_openalex_to_researchmind_schema(openalex_df: pd.DataFrame) -> pd.DataFrame:
    df = openalex_df.copy()

    if df.empty:
        return _empty_researchmind_frame()

    for col in ["title", "publication_year", "doi", "type", "source", "primary_topic", "open_access"]:
        if col not in df.columns:
            df[col] = ""

    year = pd.to_numeric(df["publication_year"], errors="coerce")

    out = pd.DataFrame({
        "pmid": "",
        "doi": df["doi"].fillna("").astype(str),
        "title": df["title"].fillna("Unknown").replace("", "Unknown").astype(str),
        "abstract": df["title"].fillna("").astype(str),
        "journal": df["source"].fillna("Unknown").replace("", "Unknown").astype(str),
        "pub_year": year.astype("Int64"),
        "pub_month": "Unknown",
        "pub_month_num": 0,
        "month_year": "Unknown-" + year.astype("Int64").astype(str),
        "authors": "",
        "authors_count": 0,
        "country": "Unknown",
        "research_type": df["type"].fillna("Unknown").replace("", "Unknown").astype(str),
        "keywords": df["primary_topic"].fillna("").astype(str),
        "major_topic": df["primary_topic"].fillna("Unknown").replace("", "Unknown").astype(str),
        "language": "Unknown",
        "open_access": df["open_access"].fillna("Unknown").astype(str),
        "source": "OpenAlex",
        "year": year.astype("Int64"),
    })

    out.loc[out["pub_year"].isna(), "month_year"] = "Unknown"

    if "openalex_id" in df.columns:
        out["openalex_id"] = df["openalex_id"]

    if "cited_by_count" in df.columns:
        out["cited_by_count"] = df["cited_by_count"]

    return out


def _empty_researchmind_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "pmid", "doi", "title", "abstract", "journal", "pub_year", "pub_month",
        "pub_month_num", "month_year", "authors", "authors_count", "country",
        "research_type", "keywords", "major_topic", "language", "open_access",
        "source", "year",
    ])


def deduplicate_researchmind(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    out["_pmid_key"] = _text_column(out, "pmid").str.strip()
    out["_doi_key"] = _text_column(out, "doi").str.lower().str.strip()
    out["_title_key"] = (
        _text_column(out, "title")
        .str.lower()
        .map(lambda value: re.sub(r"\s+", " ", value).strip())
    )

    for key in ["_pmid_key", "_doi_key"]:
        valid = out[key].ne("") & out[key].ne("unknown")
        with_key = out.loc[valid].drop_duplicates(subset=[key], keep="first")
        without_key = out.loc[~valid]
        out = pd.concat([with_key, without_key], ignore_index=True)

    title_only = (
        out["_pmid_key"].isin(["", "unknown"])
        & out["_doi_key"].isin(["", "unknown"])
        & out["_title_key"].ne("")
        & out["_title_key"].ne("unknown")
    )
    with_title_only = out.loc[title_only].drop_duplicates(subset=["_title_key"], keep="first")
    rest = out.loc[~title_only]
    out = pd.concat([rest, with_title_only], ignore_index=True)

    return out.drop(columns=["_pmid_key", "_doi_key", "_title_key"], errors="ignore")


def _text_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series("", index=df.index, dtype="object")

    return df[column].fillna("").astype(str)


def compute_top_topics(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    topic_col = "major_topic" if "major_topic" in df.columns else "research_type"

    if topic_col not in df.columns:
        return pd.DataFrame(columns=["topic", "topic_count"])

    out = (
        df[topic_col]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda series: ~series.str.lower().isin(STOP_TOPICS)]
        .value_counts()
        .head(n)
        .reset_index()
    )
    out.columns = ["topic", "topic_count"]
    return out


def compute_top_keywords(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    if "keywords" not in df.columns:
        return pd.DataFrame(columns=["keyword", "keyword_count"])

    out = split_keywords(df["keywords"]).reset_index()

    if out.empty:
        return pd.DataFrame(columns=["keyword", "keyword_count"])

    out.columns = ["keyword", "keyword_count"]
    out = out[~out["keyword"].astype(str).str.lower().isin(STOP_TOPICS)].head(n)
    return out


def run_full_analysis(config: dict) -> dict:
    source = config["data_source"]
    raw_query = config["query"]
    selected_domain = config.get("selected_domain", current_selected_domain())
    query = preprocess_research_query(raw_query)
    config = config.copy()
    config["query"] = query
    config["selected_domain"] = selected_domain
    warnings: list[str] = []
    errors: list[str] = []
    diagnostics: dict = {
        "pubmed_called": False,
        "pubmed_pmids_fetched": 0,
        "pubmed_raw_records": 0,
        "pubmed_normalized_records": 0,
        "pubmed_error": "",
        "pubmed_user_message": "",
        "pubmed_original_query": "",
        "pubmed_fallback_queries": [],
        "pubmed_successful_query": "",
        "pubmed_final_status": "",
    }
    openalex_gap = None

    df = _load_source_dataframe(config, warnings, errors, diagnostics)

    if df.empty:
        warnings.append("No records were available for analysis.")

    dedup_before_count = len(df)
    source_distribution_before = source_distribution(df)
    df = deduplicate_researchmind(df)
    dedup_after_count = len(df)
    source_distribution_after = source_distribution(df)

    publication = _safe_dataframe_call("publication trend", publication_trend, df, errors)
    semantic_mask = semantic_match_mask(df, query)
    semantic_df = df.loc[semantic_mask].copy()
    topic_df = semantic_df if not semantic_df.empty else df
    top_topics = compute_top_topics(topic_df)
    top_keywords = compute_top_keywords(topic_df)
    country = _safe_dataframe_call("country analysis", top_counts, df, "country", 15, errors)
    journal = _safe_dataframe_call("journal analysis", top_counts, df, "journal", 15, errors)
    research_type = _safe_dataframe_call(
        "research type analysis", top_counts, df, "research_type", 15, errors
    )
    strict_gap_score = _safe_dict_call("strict research gap score", research_gap_score, df, query, errors)
    query_trend = semantic_query_trend(df, query)
    gap_score = semantic_research_gap_score(df, query, strict_gap_score)
    semantic_results = _safe_dataframe_call(
        "semantic search", semantic_search_tfidf, df, query, 8, errors
    )

    if source in {"OpenAlex Live", "Hybrid: OpenAlex + PubMed"}:
        openalex_gap = _run_openalex_gap(config, warnings, errors)

        if source == "OpenAlex Live" and openalex_gap and query_trend.empty:
            publication = _as_dataframe(openalex_gap.get("trend"))
            query_trend = publication.copy()

    ai_suggestions = _generate_ai_suggestions(source, df, query, config, errors)
    domain_reasoning = build_domain_reasoning(query, ai_suggestions)
    strategic_score = compute_strategic_opportunity_score(
        gap_score,
        query,
        source_distribution_after,
        openalex_gap,
        query_trend,
        top_topics,
        top_keywords,
    )
    ai_insight = generate_ai_insight(
        gap_score,
        top_topics,
        top_keywords,
        query_trend,
        ai_suggestions,
        source_distribution_after,
        query,
    )
    research_strategy = build_research_strategy(query, ai_suggestions)
    paperability_score = build_paperability_score(
        query,
        gap_score,
        strategic_score,
        research_strategy,
        ai_suggestions,
        openalex_gap,
        source_distribution_after,
        domain_reasoning,
    )

    results = {
        "data_source": source,
        "data_source_label": config.get("data_source_label", DATA_SOURCE_LABELS.get(source, source)),
        "selected_domain": selected_domain,
        "query": query,
        "raw_query": raw_query,
        "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "normalized_dataset": df,
        "publication_trend": publication,
        "top_topics": top_topics,
        "top_keywords": top_keywords,
        "country_analysis": country,
        "journal_analysis": journal,
        "research_type_analysis": research_type,
        "query_trend": query_trend,
        "gap_score": gap_score,
        "strict_gap_score": strict_gap_score,
        "strategic_opportunity_score": strategic_score,
        "semantic_search_results": semantic_results,
        "ai_topic_suggestions": ai_suggestions,
        "ai_research_insight": ai_insight,
        "research_strategy": research_strategy,
        "domain_reasoning": domain_reasoning,
        "paperability_score": paperability_score,
        "openalex_gap": openalex_gap,
        "warnings": warnings,
        "errors": errors,
        "diagnostics": diagnostics,
        "dedup_before_count": dedup_before_count,
        "dedup_after_count": dedup_after_count,
        "source_distribution_before": source_distribution_before,
        "source_distribution": source_distribution_after,
    }
    results = apply_domain_guard_to_results(results)
    results["export_path"] = export_analysis_results(results)
    return results


def _load_source_dataframe(
    config: dict,
    warnings: list[str],
    errors: list[str],
    diagnostics: dict,
) -> pd.DataFrame:
    source = config["data_source"]

    if source == "Local CSV":
        return load_local_csv(config["csv_path"], config["row_limit"])

    if source == "PubMed Live":
        return _load_pubmed_dataframe(
            config["query"],
            config["pubmed_max_results"],
            config["years_back"],
            config.get("pubmed_email", ""),
            config.get("pubmed_api_key", ""),
            warnings,
            errors,
            diagnostics,
        )

    if source == "OpenAlex Live":
        openalex_gap = _run_openalex_gap(config, warnings, errors)
        config["_openalex_gap"] = openalex_gap
        return _openalex_gap_to_dataframe(openalex_gap)

    if source == "Hybrid: OpenAlex + PubMed":
        pubmed_df = _load_pubmed_dataframe(
            config["query"],
            config["pubmed_max_results"],
            config["years_back"],
            config.get("pubmed_email", ""),
            config.get("pubmed_api_key", ""),
            warnings,
            errors,
            diagnostics,
        )
        openalex_gap = _run_openalex_gap(config, warnings, errors)
        config["_openalex_gap"] = openalex_gap
        openalex_df = _openalex_gap_to_dataframe(openalex_gap)

        if pubmed_df.empty and not openalex_df.empty and diagnostics.get("pubmed_error"):
            warnings.append(
                "Hibrit analizde PubMed bağlantısı başarısız oldu; OpenAlex sonuçlarıyla devam edildi."
            )
        elif pubmed_df.empty and not openalex_df.empty and diagnostics.get("pubmed_final_status") == "no_results":
            warnings.append(
                "Hibrit analizde PubMed bu spesifik konuda sonuç döndürmedi; OpenAlex sonuçlarıyla devam edildi."
            )

        return pd.concat([pubmed_df, openalex_df], ignore_index=True)

    errors.append(f"Unknown data source: {source}")
    return _empty_researchmind_frame()


def _load_pubmed_dataframe(
    query: str,
    max_results: int,
    years_back: int,
    email: str,
    api_key: str,
    warnings: list[str],
    errors: list[str],
    diagnostics: dict,
) -> pd.DataFrame:
    diagnostics["pubmed_called"] = True
    diagnostics["pubmed_original_query"] = query
    candidate_queries = pubmed_fallback_queries(query)
    diagnostics["pubmed_fallback_queries"] = candidate_queries[1:]
    last_error = ""

    env_config = get_pubmed_config()
    config = PubMedConfig(
        api_key=str(api_key).strip() or env_config.api_key,
        email=str(email).strip() or env_config.email,
    )
    client = PubMedClient(config=config)

    for candidate in candidate_queries:
        attempts = 3
        for attempt in range(attempts):
            try:
                pmids = client.esearch(
                    query=candidate,
                    max_results=int(max_results),
                    years_back=int(years_back),
                )
                search_metadata = getattr(client, "last_search_metadata", {}) or {}
                tried_queries = search_metadata.get("tried_queries") or []
                if tried_queries:
                    combined = [*diagnostics.get("pubmed_fallback_queries", []), *tried_queries[1:]]
                    diagnostics["pubmed_fallback_queries"] = list(dict.fromkeys(combined))
                diagnostics["pubmed_final_status"] = search_metadata.get("final_status", "")
                diagnostics["pubmed_pmids_fetched"] = len(pmids)

                if not pmids:
                    if search_metadata.get("service_unavailable"):
                        last_error = str(search_metadata.get("error") or "PubMed service unavailable.")
                        diagnostics["pubmed_error"] = last_error
                        diagnostics["pubmed_final_status"] = "service_unavailable"
                    break

                records = client.efetch(pmids)
                diagnostics["pubmed_raw_records"] = len(records)
                normalized = normalize_pubmed_to_researchmind_schema(pd.DataFrame(records))
                diagnostics["pubmed_normalized_records"] = len(normalized)
                diagnostics["pubmed_successful_query"] = search_metadata.get("successful_query") or candidate
                diagnostics["pubmed_final_status"] = "success"
                return normalized
            except PubMedClientError as exc:
                last_error = exc.user_message
                diagnostics["pubmed_final_status"] = "service_unavailable" if is_transient_pubmed_error(last_error) else "failed"
                if is_transient_pubmed_error(last_error) and attempt < attempts - 1:
                    time.sleep(0.8 * (attempt + 1))
                    continue
                break
            except Exception as exc:
                last_error = str(exc)
                diagnostics["pubmed_final_status"] = "service_unavailable" if is_transient_pubmed_error(last_error) else "failed"
                if is_transient_pubmed_error(last_error) and attempt < attempts - 1:
                    time.sleep(0.8 * (attempt + 1))
                    continue
                break

    if not last_error:
        diagnostics["pubmed_error"] = ""
        diagnostics["pubmed_final_status"] = "no_results"
        diagnostics["pubmed_user_message"] = "PubMed araması bu spesifik konuda sonuç döndürmedi; OpenAlex ile analiz tamamlandı."
        warnings.append(diagnostics["pubmed_user_message"])
        return _empty_researchmind_frame()

    diagnostics["pubmed_error"] = last_error
    diagnostics["pubmed_final_status"] = diagnostics.get("pubmed_final_status") or "service_unavailable"
    diagnostics["pubmed_user_message"] = "PubMed geçici olarak yanıt vermedi. OpenAlex sonuçlarıyla analiz tamamlandı."
    errors.append(diagnostics["pubmed_user_message"])
    return _empty_researchmind_frame()


def _run_openalex_gap(config: dict, warnings: list[str], errors: list[str]) -> dict | None:
    if "_openalex_gap" in config:
        return config["_openalex_gap"]

    openalex_contact = str(config.get("openalex_api_key", "")).strip() or get_config_value("OPENALEX_EMAIL")

    if not openalex_contact:
        warnings.append("OpenAlex e-posta veya API bilgisi eklenmemiş. Yoğun kullanımda erişim sınırlanabilir.")

    try:
        return openalex_gap_analysis(
            query=config["query"],
            api_key=openalex_contact,
            per_page=int(config.get("openalex_max_results", DEFAULT_LIVE_RESULT_LIMIT)),
            years_back=int(config.get("years_back", 5)),
        )
    except Exception as exc:
        errors.append(f"OpenAlex live gap analysis failed: {exc}")
        return None


def _openalex_gap_to_dataframe(openalex_gap: dict | None) -> pd.DataFrame:
    if not openalex_gap:
        return _empty_researchmind_frame()

    works = _as_dataframe(openalex_gap.get("results"))
    trend = _as_dataframe(openalex_gap.get("trend"))

    if not works.empty and "publication_year" in works.columns and "publication_year" in trend.columns:
        years = pd.to_numeric(trend["publication_year"], errors="coerce").dropna()

        if not years.empty:
            publication_year = pd.to_numeric(works["publication_year"], errors="coerce")
            works = works.loc[publication_year.between(int(years.min()), int(years.max()))].copy()

    return normalize_openalex_to_researchmind_schema(works)


def _generate_ai_suggestions(
    source: str,
    df: pd.DataFrame,
    query: str,
    config: dict,
    errors: list[str],
) -> pd.DataFrame:
    try:
        if source in {"OpenAlex Live", "Hybrid: OpenAlex + PubMed"}:
            openalex_contact = str(config.get("openalex_api_key", "")).strip() or get_config_value("OPENALEX_EMAIL")
            suggestions = generate_ai_research_topic_suggestions(
                query=query,
                api_key=openalex_contact,
                years_back=int(config.get("years_back", 5)),
            )
            suggestions = apply_domain_reasoning_filter(suggestions, query)
            return naturalize_suggestions(suggestions, query)

        suggestions = suggest_research_opportunities(df, query, top_n=8)
        suggestions = apply_domain_reasoning_filter(suggestions, query)
        return naturalize_suggestions(suggestions, query)
    except Exception as exc:
        errors.append(f"AI topic suggestions failed: {exc}")
        return pd.DataFrame()


def _safe_dataframe_call(label: str, func, *args) -> pd.DataFrame:
    errors = args[-1]
    call_args = args[:-1]

    try:
        return _as_dataframe(func(*call_args))
    except Exception as exc:
        errors.append(f"{label} failed: {exc}")
        return pd.DataFrame()


def _safe_dict_call(label: str, func, *args) -> dict:
    errors = args[-1]
    call_args = args[:-1]

    try:
        value = func(*call_args)
        return value if isinstance(value, dict) else {}
    except Exception as exc:
        errors.append(f"{label} failed: {exc}")
        return {}


def _as_dataframe(value) -> pd.DataFrame:
    return value if isinstance(value, pd.DataFrame) else pd.DataFrame()


def _register_pdf_font() -> str:
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    except Exception:
        return "Helvetica"

    font_candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/Library/Fonts/Arial Unicode.ttf"),
    ]

    for font_path in font_candidates:
        if not font_path.exists():
            continue

        try:
            pdfmetrics.registerFont(TTFont("ResearchMindUnicode", str(font_path)))
            return "ResearchMindUnicode"
        except Exception:
            continue

    return "Helvetica"


def generate_pdf_report(summary_text: str, output_path: str | Path) -> bool:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import cm
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    except Exception:
        return False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    font_name = _register_pdf_font()

    try:
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "ResearchMindTitle",
            parent=styles["Title"],
            fontName=font_name,
            fontSize=20,
            leading=24,
            textColor=colors.HexColor("#15324a"),
            spaceAfter=14,
        )
        section_style = ParagraphStyle(
            "ResearchMindSection",
            parent=styles["Heading2"],
            fontName=font_name,
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#1f6f8b"),
            spaceBefore=10,
            spaceAfter=5,
        )
        body_style = ParagraphStyle(
            "ResearchMindBody",
            parent=styles["BodyText"],
            fontName=font_name,
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#222222"),
            spaceAfter=4,
        )
        bullet_style = ParagraphStyle(
            "ResearchMindBullet",
            parent=body_style,
            leftIndent=12,
            firstLineIndent=-8,
        )

        story = [
            Paragraph("ResearchMind AI Analysis Report", title_style),
            Paragraph("SCI Publication Potential and Research Intelligence Summary", body_style),
            Spacer(1, 0.25 * cm),
        ]

        for raw_line in str(summary_text or "").splitlines():
            line = raw_line.strip()

            if not line:
                story.append(Spacer(1, 0.12 * cm))
                continue

            clean = html.escape(line)
            if line.endswith(":") and not line.startswith("-"):
                story.append(Paragraph(clean, section_style))
            elif line.startswith("-"):
                story.append(Paragraph(clean, bullet_style))
            elif line == "ResearchMind AI Özet Raporu":
                continue
            else:
                story.append(Paragraph(clean, body_style))

        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            rightMargin=1.6 * cm,
            leftMargin=1.6 * cm,
            topMargin=1.5 * cm,
            bottomMargin=1.5 * cm,
            title="ResearchMind AI Analysis Report",
        )
        doc.build(story)
        return output_path.exists()
    except Exception:
        return False


def _score_band(score) -> tuple[str, str]:
    value = parse_numeric(score)
    if value >= 80:
        return "Very Strong", "high"
    if value >= 65:
        return "Strong", "high"
    if value >= 40:
        return "Competitive", "mid"
    return "Saturated", "low"


def build_executive_summary(results: dict) -> str:
    gap = results.get("gap_score") or {}
    domain = results.get("domain_reasoning") or {}
    paperability = results.get("paperability_score") or {}
    strategic = parse_numeric(results.get("strategic_opportunity_score"))
    paper_score = parse_numeric(paperability.get("total_score"))
    matched = parse_numeric(gap.get("total_records"))
    growth = parse_numeric(gap.get("growth_rate"))
    clinical_domain = domain.get("clinical_domain", "interdisciplinary applied AI")
    if not clinical_domain or str(clinical_domain).lower() == "not detected":
        clinical_domain = "interdisciplinary applied AI"
    modality = domain.get("primary_modality", "the available evidence base")

    if matched > 250:
        competition = "highly competitive"
    elif matched > 50:
        competition = "moderately competitive"
    elif matched > 10:
        competition = "an emerging but already visible"
    else:
        competition = "early-stage and comparatively underexplored"

    growth_phrase = "with a positive recent growth signal" if growth > 0 else "with limited recent growth evidence"
    opportunity_phrase = "strategically differentiable" if strategic >= 60 else "requiring sharper narrowing"
    publication_phrase = "strong publication potential" if paper_score >= 65 else "moderate publication potential"
    diagnostics = results.get("diagnostics", {}) or {}
    pubmed_note = ""
    if diagnostics.get("pubmed_final_status") == "no_results":
        pubmed_note = " PubMed returned no results for this specific query; OpenAlex was used for the analysis."
    elif diagnostics.get("pubmed_final_status") == "service_unavailable":
        pubmed_note = " PubMed temporarily unavailable; OpenAlex results were used for analysis."

    return (
        f"This topic represents {competition} and {opportunity_phrase} research area within {clinical_domain}. "
        f"The current evidence landscape is primarily shaped by {modality}, {growth_phrase}. "
        f"Based on the combined opportunity, domain consistency, and paperability signals, the topic shows "
        f"{publication_phrase}, provided the study is framed around a specific method, validation strategy, "
        f"and clinically meaningful differentiation.{pubmed_note}"
    )


def is_generic_paper_title(title: str, query: str) -> bool:
    key = normalize_topic_key(title)
    if not key:
        return True

    generic_suffixes = {
        "early diagnosis", "pet", "mri", "biomarker prediction", "transformer model",
        "medical imaging", "explainable ai", "multimodal learning", "diagnosis",
        "classification", "deep learning", "machine learning",
    }
    if key.startswith("ai based ") and key.replace("ai based ", "", 1).strip() in generic_suffixes:
        return True

    concepts = extract_query_concepts(query)
    title_concepts = extract_query_concepts(title)
    has_domain = bool(set(concepts.get("disease", [])) & set(title_concepts.get("disease", [])))
    has_clinical_domain = bool(set(concepts.get("clinical_domain", [])) & set(title_concepts.get("clinical_domain", [])))
    has_modality = (
        bool(set(concepts.get("modality", [])) & set(title_concepts.get("modality", [])))
        if concepts.get("modality") else bool(title_concepts.get("modality"))
    )
    has_method = (
        bool(set(concepts.get("method", [])) & set(title_concepts.get("method", [])))
        if concepts.get("method") else bool(title_concepts.get("method"))
    )
    has_task = (
        bool(set(concepts.get("task", [])) & set(title_concepts.get("task", [])))
        if concepts.get("task") else bool(title_concepts.get("task"))
    )

    query_has_domain = bool(concepts.get("disease") or concepts.get("clinical_domain"))
    if query_has_domain and not has_domain and not has_clinical_domain:
        return True

    return not (has_modality or has_method) or not (has_task or has_domain or has_clinical_domain)


def synthesize_paper_titles(results: dict, min_count: int = 5) -> list[str]:
    query = results.get("query", "")
    selected_domain = results.get("selected_domain", current_selected_domain())
    concepts = extract_query_concepts(query)
    disease = concepts.get("disease", [])
    modality = concepts.get("modality", [])
    method = concepts.get("method", [])
    suggestions = _as_dataframe(results.get("ai_topic_suggestions"))
    titles = []
    suggestion_titles = []

    topic_col = "suggested_research_topic" if "suggested_research_topic" in suggestions.columns else "suggested_topic"
    if topic_col in suggestions.columns:
        suggestion_titles.extend(suggestions[topic_col].dropna().astype(str).head(3).tolist())

    if selected_domain == ENGINEERING_DOMAIN and not is_engineering_health_hybrid(query):
        titles.extend([item["title"] for item in engineering_topic_refinement(query)])
    elif "autism" in disease:
        titles.extend([
            "Federated Explainable EEG-Eye Tracking Fusion for Early Autism Detection",
            "Privacy-Aware Multimodal Neurodevelopmental AI for ASD Screening",
            "Explainable Temporal Transformers for EEG-Based Autism Screening",
            "Gaze-Assisted Multimodal AI for Early Autism Detection",
            "Domain-Consistent Federated Learning for Neurodevelopmental Screening",
        ])
    elif "alzheimer" in disease:
        titles.extend([
            "Explainable Multimodal MRI-PET Fusion for Early Alzheimer Diagnosis",
            "Federated Neuroimaging AI for Privacy-Preserving Alzheimer Detection",
            "Transformer-Based Explainable AI for Alzheimer MRI Analysis",
            "Clinically Validated Vision Transformers for Dementia Neuroimaging",
            "Privacy-Aware Multimodal Learning for Alzheimer Progression Modeling",
        ])
    elif "breast cancer" in disease:
        titles.extend([
            "Explainable CNN-Based Histopathology Analysis for Breast Cancer Classification",
            "Federated Mammography AI for Privacy-Preserving Breast Cancer Screening",
            "Attention-Enhanced CNN Models for Breast Cancer Image Classification",
            "External Validation of Explainable Deep Learning for Breast Cancer Detection",
            "Lightweight CNN Decision Support for Breast Cancer Mammography Screening",
        ])
    elif "depression" in disease and "eeg" in modality:
        titles.extend([
            "Spectral Attention Networks for EEG-Based Depression Detection",
            "Wavelet-CNN Fusion for Robust Depression Classification from EEG",
            "Explainable Temporal Transformers for Depression Biosignal Screening",
            "Subject-Level Validation of EEG Deep Learning Models for Depression",
            "Privacy-Aware EEG Analytics for Clinical Depression Detection",
        ])
    elif "blockchain" in method:
        titles.extend([
            "Blockchain-Based Privacy and Interoperability Framework for Healthcare Data Security",
            "Smart Contract-Enabled Clinical Data Sharing with Privacy-Aware Validation",
            "Decentralized Healthcare Data Integrity Framework Using Blockchain",
            "Auditability-Centered Blockchain Architecture for Secure Medical Records",
            "Interoperable Healthcare Data Governance Using Blockchain Smart Contracts",
        ])
    elif "sports medicine" in concepts.get("clinical_domain", []) or "sports injury" in disease:
        titles.extend([
            "Explainable Deep Learning for Football Player Injury Risk Prediction",
            "Wearable Sensor-Based Injury Risk Assessment in Professional Football",
            "Temporal Transformer Models for Training Load and Injury Prediction in Soccer",
            "Interpretable Player Workload Modeling for Football Injury Prevention",
            "Multi-Season Validation of AI-Based Injury Risk Scoring in Football Players",
        ])
    else:
        base = title_case_topic(query)
        titles.extend([
            f"Explainable AI Framework for {base}",
            f"Clinical Validation Strategy for {base}",
            f"Robust and Interpretable Learning for {base}",
            f"Dataset-Centered Evaluation of {base}",
            f"Decision-Support Modeling for {base}",
        ])

    titles.extend(suggestion_titles)

    clean_titles = []
    seen = set()
    for title in titles:
        clean = re.sub(r"\s+", " ", str(title or "")).strip(" .")
        key = normalize_topic_key(clean)
        if clean and key not in seen and (selected_domain == ENGINEERING_DOMAIN or not is_generic_paper_title(clean, query)):
            seen.add(key)
            clean_titles.append(clean)
        if len(clean_titles) >= min_count:
            break
    if len(clean_titles) < min_count:
        fallback_titles = domain_adapted_suggestions(query)
        if "suggested_research_topic" in fallback_titles.columns:
            for title in fallback_titles["suggested_research_topic"].dropna().astype(str).tolist():
                key = normalize_topic_key(title)
                if key not in seen and (selected_domain == ENGINEERING_DOMAIN or not is_generic_paper_title(title, query)):
                    seen.add(key)
                    clean_titles.append(title)
                if len(clean_titles) >= min_count:
                    break
    return clean_titles


def build_opportunity_analysis(results: dict) -> dict[str, str]:
    gap = results.get("gap_score") or {}
    paperability = results.get("paperability_score") or {}
    matched = parse_numeric(gap.get("total_records"))
    strategic = parse_numeric(results.get("strategic_opportunity_score"))
    paper_score = parse_numeric(paperability.get("total_score"))
    domain = results.get("domain_reasoning") or {}

    competition = (
        "The field appears crowded and requires a narrow claim." if matched > 250
        else "The field is competitive but still differentiable with a focused method." if matched > 50
        else "The field has manageable competition and room for positioning."
    )
    novelty = (
        "Novelty is strongest when the topic is framed around a specific clinical validation and method combination."
        if strategic >= 60 else
        "Novelty should be strengthened by narrowing the disease, modality, and validation protocol."
    )
    clinical = f"The clinical context is anchored in {domain.get('clinical_domain', 'the target domain')}, which supports practical relevance."
    method = "Methodological strength is supported by explainability, multimodal/federated design, or evidence-specific modeling." if paper_score >= 60 else "Methodological strength should be increased through stronger baselines and validation."
    dataset = "Dataset feasibility depends on access to modality-specific evidence and external validation cohorts."

    return {
        "Competition Level": competition,
        "Novelty Potential": novelty,
        "Clinical Relevance": clinical,
        "Methodological Strength": method,
        "Dataset Feasibility": dataset,
    }


def final_strategic_recommendation(results: dict) -> str:
    paperability = results.get("paperability_score") or {}
    domain = results.get("domain_reasoning") or {}
    score = parse_numeric(paperability.get("total_score"))
    consistency = domain.get("domain_consistency", "Medium")

    if score >= 70 and consistency == "High":
        return (
            "This topic should be pursued, but with a deliberately narrow research claim. "
            "A focus on explainable multimodal modeling, privacy-aware validation, and external clinical evidence "
            "would improve differentiation compared with conventional single-modality AI studies."
        )
    if score >= 50:
        return (
            "This topic is worth pursuing after refinement. The strongest path is to reduce scope, define a precise "
            "dataset and validation plan, and position the contribution against the most saturated methodological baseline."
        )
    return (
        "This topic should be reframed before execution. The current signals suggest that publication potential depends "
        "on stronger domain alignment, clearer novelty, and a more defensible validation strategy."
    )


def generate_executive_pdf_report(results: dict, output_path: str | Path, summary_text: str = "") -> bool:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import HRFlowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except Exception:
        return generate_pdf_report(summary_text, output_path)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    font_name = _register_pdf_font()

    try:
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            rightMargin=1.35 * cm,
            leftMargin=1.35 * cm,
            topMargin=1.25 * cm,
            bottomMargin=1.25 * cm,
            title="ResearchMind AI Executive Report",
        )
        width = A4[0] - doc.leftMargin - doc.rightMargin

        title = ParagraphStyle("ExecTitle", fontName=font_name, fontSize=28, leading=33, textColor=colors.white, spaceAfter=12)
        subtitle = ParagraphStyle("ExecSubtitle", fontName=font_name, fontSize=14, leading=18, textColor=colors.HexColor("#c7d2fe"))
        h1 = ParagraphStyle("ExecH1", fontName=font_name, fontSize=17, leading=21, textColor=colors.HexColor("#12324a"), spaceBefore=8, spaceAfter=8)
        h2 = ParagraphStyle("ExecH2", fontName=font_name, fontSize=11, leading=14, textColor=colors.HexColor("#2563eb"), spaceAfter=4)
        body = ParagraphStyle("ExecBody", fontName=font_name, fontSize=9.3, leading=13.2, textColor=colors.HexColor("#243447"))
        small = ParagraphStyle("ExecSmall", fontName=font_name, fontSize=8.2, leading=11, textColor=colors.HexColor("#475569"))
        white_small = ParagraphStyle("ExecWhiteSmall", fontName=font_name, fontSize=9, leading=12, textColor=colors.HexColor("#e5edf6"))

        def p(text, style=body):
            clean = html.escape(str(text or "-")).replace("\n", "<br/>")
            clean = clean.replace("&lt;br/&gt;", "<br/>").replace("&lt;br /&gt;", "<br/>")
            return Paragraph(clean, style)

        def card(title_text, value_text, note_text="", accent="#2563eb"):
            return Table(
                [[p(title_text, h2)], [p(value_text, ParagraphStyle("CardValue", fontName=font_name, fontSize=18, leading=22, textColor=colors.HexColor(accent)))], [p(note_text, small)]],
                colWidths=[width / 3 - 8],
                style=TableStyle([
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                    ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#dbeafe")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ]),
            )

        df = _as_dataframe(results.get("normalized_dataset"))
        distribution = results.get("source_distribution", {})
        gap = results.get("gap_score") or {}
        strategy = results.get("research_strategy") or {}
        domain = results.get("domain_reasoning") or {}
        paperability = results.get("paperability_score") or {}
        strategic = results.get("strategic_opportunity_score", gap.get("gap_score", "-"))
        paper_score = paperability.get("total_score", "-")
        strategic_label, strategic_color = _score_band(strategic)
        paper_label, paper_color = _score_band(paper_score)
        color_map = {"high": "#16a34a", "mid": "#d97706", "low": "#dc2626"}

        source_text = f"PubMed: {distribution.get('PubMed', 0)} | OpenAlex: {distribution.get('OpenAlex', 0)} | Local: {distribution.get('Yerel', 0)}"
        cover_table = Table(
            [[
                p("ResearchMind AI", title),
                p(
                    "Research Intelligence Report<br/><br/>"
                    f"Research Topic: {results.get('query', '-')}<br/>"
                    f"Selected Research Domain: {results.get('selected_domain', '-')}<br/>"
                    f"Analysis Date: {results.get('analysis_time', '-')}<br/>"
                    f"Data Sources: {source_text}<br/>"
                    f"Strategic Opportunity Score: {strategic}<br/>"
                    f"Paperability Score: {paper_score}",
                    subtitle,
                ),
            ]],
            colWidths=[width * 0.45, width * 0.55],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#0f172a")),
                ("BOX", (0, 0), (-1, -1), 1.2, colors.HexColor("#38bdf8")),
                ("LEFTPADDING", (0, 0), (-1, -1), 20),
                ("RIGHTPADDING", (0, 0), (-1, -1), 20),
                ("TOPPADDING", (0, 0), (-1, -1), 28),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 28),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]),
        )

        story = [cover_table, Spacer(1, 0.6 * cm), HRFlowable(width="100%", color=colors.HexColor("#38bdf8")), Spacer(1, 0.35 * cm)]
        story.append(p("Executive Summary", h1))
        story.append(p(build_executive_summary(results), body))
        story.append(PageBreak())

        story.append(p("Research Intelligence KPIs", h1))
        kpi_rows = [
            [
                card("Total Records", f"{len(df):,}", "Normalized corpus size"),
                card("Semantic Match Count", gap.get("total_records", 0), "Topic-aware match volume"),
                card("Strategic Opportunity", strategic, strategic_label, color_map[strategic_color]),
            ],
            [
                card("Paperability Score", paper_score, paperability.get("level_tr", paper_label), color_map[paper_color]),
                card("Domain Consistency", domain.get("domain_consistency", "-"), f"Score: {domain.get('domain_consistency_score', '-')}", "#2563eb"),
                card("Growth Rate", gap.get("growth_rate", "-"), "Recent momentum signal"),
            ],
        ]
        story.append(Table(kpi_rows, colWidths=[width / 3] * 3, style=TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("BOTTOMPADDING", (0, 0), (-1, -1), 10)])))

        story.append(p("AI Research Insight", h1))
        story.append(Table([[p(results.get("ai_research_insight", "Insight could not be generated."), body)]], colWidths=[width], style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#ecfeff")),
            ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#67e8f9")),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ])))

        story.append(p("Research Strategy Recommendations", h1))
        strategy_rows = [
            [p("Recommended Methodology", h2), p(strategy.get("methodology", "-"), body)],
            [p("Suggested Validation Strategy", h2), p("Prioritize external validation, clinically meaningful endpoints, and transparent error analysis.", body)],
            [p("Dataset / Evidence Focus", h2), p(strategy.get("evidence", "-"), body)],
            [p("Differentiation Strategy", h2), p(strategy.get("differentiation", "-"), body)],
        ]
        story.append(Table(strategy_rows, colWidths=[width * 0.32, width * 0.68], style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#e2e8f0")),
            ("LEFTPADDING", (0, 0), (-1, -1), 9),
            ("RIGHTPADDING", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ])))

        story.append(p("Domain Reasoning", h1))
        domain_rows = [
            [p("Selected Research Domain", h2), p(results.get("selected_domain", "-"), body)],
            [p("Primary Domain", h2), p(domain.get("clinical_domain", "-"), body)],
            [p("Primary Modality", h2), p(domain.get("primary_modality", "-"), body)],
            [p("Dominant Methodology", h2), p(domain.get("primary_method", "-"), body)],
            [p("Domain Consistency", h2), p(f"{domain.get('domain_consistency', '-')} ({domain.get('domain_consistency_score', '-')})", body)],
        ]
        story.append(Table(domain_rows, colWidths=[width * 0.32, width * 0.68], style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0f9ff")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bae6fd")),
            ("LEFTPADDING", (0, 0), (-1, -1), 9),
            ("RIGHTPADDING", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ])))

        story.append(PageBreak())
        story.append(p("Suggested Paper Titles", h1))
        for index, title_text in enumerate(synthesize_paper_titles(results, min_count=5), start=1):
            story.append(Table([[p(f"{index:02d}", ParagraphStyle("TitleIndex", fontName=font_name, fontSize=15, leading=18, textColor=colors.HexColor("#2563eb"))), p(title_text, body)]], colWidths=[1.2 * cm, width - 1.2 * cm], style=TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#dbeafe")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ])))
            story.append(Spacer(1, 0.08 * cm))

        story.append(p("Paperability Score", h1))
        strengths = paperability.get("reasons", [])[:3] or ["Domain-aligned research positioning."]
        risks = []
        matched = parse_numeric(gap.get("total_records"))
        if matched > 50:
            risks.append("Competition is meaningful; the contribution must be narrowed.")
        if domain.get("semantic_leakage_risk") in {"Medium", "High"}:
            risks.append("Domain leakage risk should be monitored during literature framing.")
        risks.append("Dataset access and external validation may determine publication readiness.")
        potential_rows = [
            [p("Total Score", h2), p(str(paper_score), body)],
            [p("Publication Potential Level", h2), p(paperability.get("level_tr", paper_label), body)],
            [p("Strongest Strengths", h2), p("<br/>".join(f"- {html.escape(item)}" for item in strengths), body)],
            [p("Main Risks", h2), p("<br/>".join(f"- {html.escape(item)}" for item in risks[:3]), body)],
            [p("Recommended Narrowing Direction", h2), p(paperability.get("recommended_next_action", "-"), body)],
        ]
        story.append(Table(potential_rows, colWidths=[width * 0.32, width * 0.68], style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff7ed")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#fed7aa")),
            ("LEFTPADDING", (0, 0), (-1, -1), 9),
            ("RIGHTPADDING", (0, 0), (-1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ])))

        story.append(p("Final Strategic Recommendation", h1))
        story.append(Table([[p("Should this topic be pursued?", h2)], [p(final_strategic_recommendation(results), body)]], colWidths=[width], style=TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#eef2ff")),
            ("BOX", (0, 0), (-1, -1), 0.9, colors.HexColor("#818cf8")),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ])))
        story.append(Spacer(1, 0.25 * cm))
        story.append(p("Estimated decision-support report. This output is not a publication guarantee.", small))

        doc.build(story)
        return output_path.exists()
    except Exception:
        return generate_pdf_report(summary_text, output_path)


def export_analysis_results(results: dict) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    export_dir = OUTPUTS_DIR / timestamp
    export_dir.mkdir(parents=True, exist_ok=True)

    dataframe_exports = {
        "normalized_dataset.csv": results.get("normalized_dataset"),
        "publication_trend.csv": results.get("publication_trend"),
        "top_topics.csv": results.get("top_topics"),
        "top_keywords.csv": results.get("top_keywords"),
        "country_analysis.csv": results.get("country_analysis"),
        "journal_analysis.csv": results.get("journal_analysis"),
        "research_type_analysis.csv": results.get("research_type_analysis"),
        "query_trend.csv": results.get("query_trend"),
        "semantic_search_results.csv": results.get("semantic_search_results"),
        "ai_topic_suggestions.csv": results.get("ai_topic_suggestions"),
        "paperability_score.csv": paperability_to_dataframe(results.get("paperability_score")),
        "domain_reasoning.csv": domain_reasoning_to_dataframe(results.get("domain_reasoning")),
        "domain_guard.csv": domain_reasoning_to_dataframe(results.get("domain_guard")),
    }

    for filename, value in dataframe_exports.items():
        df = _as_dataframe(value)
        df.to_csv(export_dir / filename, index=False, encoding="utf-8-sig")

    gap_score = results.get("gap_score") or {}
    pd.DataFrame([gap_score]).to_csv(
        export_dir / "gap_score_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = build_summary_report(results, export_dir)
    summary_path = export_dir / "summary_report.txt"
    summary_path.write_text(summary, encoding="utf-8-sig")
    generate_pdf_report(summary, export_dir / "ResearchMind_AI_Report.pdf")
    generate_executive_pdf_report(
        results,
        export_dir / f"ResearchMind_AI_Executive_Report_{timestamp}.pdf",
        summary,
    )
    return str(export_dir.resolve())


def build_summary_report(results: dict, export_dir: Path) -> str:
    df = _as_dataframe(results.get("normalized_dataset"))
    gap_score = results.get("gap_score") or {}
    suggestions = _as_dataframe(results.get("ai_topic_suggestions"))

    suggestion_lines = []
    if not suggestions.empty:
        topic_col = (
            "suggested_research_topic"
            if "suggested_research_topic" in suggestions.columns
            else "suggested_topic"
        )

        if topic_col in suggestions.columns:
            suggestion_lines = [
                f"- {value}"
                for value in suggestions[topic_col].dropna().astype(str).head(5).tolist()
            ]

    if not suggestion_lines:
        suggestion_lines = ["- Öneri üretilemedi."]

    warnings = results.get("warnings", [])
    errors = results.get("errors", [])
    diagnostics = results.get("diagnostics", {})
    distribution = results.get("source_distribution", {})
    distribution_before = results.get("source_distribution_before", {})
    paperability = results.get("paperability_score") or {}
    domain_reasoning = results.get("domain_reasoning") or {}
    domain_guard = results.get("domain_guard") or {}
    paperability_reasons = [
        f"- {item}"
        for item in paperability.get("reasons", [])
    ] or ["- Gerekçe üretilemedi."]

    warning_lines = [f"- UYARI: {item}" for item in warnings]
    error_lines = [f"- HATA: {item}" for item in errors]
    issue_lines = warning_lines + error_lines or ["- Uyarı veya hata yok."]

    return "\n".join([
        "ResearchMind AI Özet Raporu",
        "",
        f"Veri kaynağı: {results.get('data_source_label', results.get('data_source', '-'))}",
        f"Selected research domain: {results.get('selected_domain', '-')}",
        f"Ham araştırma konusu: {results.get('raw_query', results.get('query', '-'))}",
        f"Araştırma konusu: {results.get('query', '-')}",
        f"Kayıt sayısı: {len(df):,}",
        f"Analiz zamanı: {results.get('analysis_time', '-')}",
        "",
        "Veri kaynağı dağılımı:",
        f"- PubMed: {distribution.get('PubMed', 0)} kayıt",
        f"- OpenAlex: {distribution.get('OpenAlex', 0)} kayıt",
        f"- Yerel: {distribution.get('Yerel', 0)} kayıt",
        "",
        "Dedup özeti:",
        f"- Dedup öncesi satır sayısı: {results.get('dedup_before_count', len(df))}",
        f"- Dedup sonrası satır sayısı: {results.get('dedup_after_count', len(df))}",
        f"- Dedup öncesi PubMed: {distribution_before.get('PubMed', 0)}",
        f"- Dedup öncesi OpenAlex: {distribution_before.get('OpenAlex', 0)}",
        f"- Dedup öncesi Yerel: {distribution_before.get('Yerel', 0)}",
        "",
        "PubMed durumu:",
        f"- PubMed çağrıldı mı: {'Evet' if diagnostics.get('pubmed_called') else 'Hayır'}",
        f"- PubMed service unavailable: {'Evet' if diagnostics.get('pubmed_final_status') == 'service_unavailable' else 'Hayır'}",
        f"- Final status: {diagnostics.get('pubmed_final_status') or '-'}",
        f"- Original PubMed query: {diagnostics.get('pubmed_original_query') or '-'}",
        f"- Tried fallback queries: {', '.join(diagnostics.get('pubmed_fallback_queries') or []) or '-'}",
        f"- Successful PubMed query: {diagnostics.get('pubmed_successful_query') or '-'}",
        f"- Çekilen PMID: {diagnostics.get('pubmed_pmids_fetched', 0)}",
        f"- EFetch kayıt sayısı: {diagnostics.get('pubmed_raw_records', 0)}",
        f"- Normalize edilen kayıt: {diagnostics.get('pubmed_normalized_records', 0)}",
        f"- PubMed hata: {diagnostics.get('pubmed_error') or 'Yok'}",
        "",
        "Research Gap Score özeti:",
        f"- Eşleşen kayıt: {gap_score.get('total_records', '-')}",
        f"- Eski strict eşleşen kayıt: {gap_score.get('strict_matched_records', results.get('strict_gap_score', {}).get('total_records', '-'))}",
        f"- Eşleştirme yöntemi: {gap_score.get('matching_method', '-')}",
        f"- Büyüme oranı: {gap_score.get('growth_rate', '-')}",
        f"- Gap score: {gap_score.get('gap_score', '-')}",
        f"- Strategic Opportunity Score: {results.get('strategic_opportunity_score', '-')}",
        f"- Yorum: {gap_score.get('interpretation', '-')}",
        "",
        "AI Research Insight:",
        results.get("ai_research_insight", "Insight üretilemedi."),
        "",
        "Research Strategy Engine:",
        f"- Suggested Research Direction: {results.get('research_strategy', {}).get('direction', '-')}",
        f"- Recommended Methodology: {results.get('research_strategy', {}).get('methodology', '-')}",
        f"- Suggested Dataset / Evidence Focus: {results.get('research_strategy', {}).get('evidence', '-')}",
        f"- Differentiation Strategy: {results.get('research_strategy', {}).get('differentiation', '-')}",
        "",
        "Domain Reasoning:",
        f"- Selected research domain: {results.get('selected_domain', '-')}",
        f"- Primary disease: {domain_reasoning.get('primary_disease', '-')}",
        f"- Modality: {domain_reasoning.get('primary_modality', '-')}",
        f"- Clinical domain: {domain_reasoning.get('clinical_domain', '-')}",
        f"- Dominant methodology: {domain_reasoning.get('primary_method', '-')}",
        f"- Domain consistency: {domain_reasoning.get('domain_consistency', '-')} ({domain_reasoning.get('domain_consistency_score', '-')})",
        f"- Leakage risk: {domain_reasoning.get('semantic_leakage_risk', '-')}",
        f"- Filtered leakage suggestions: {domain_reasoning.get('leakage_filtered_count', 0)}",
        f"- DomainGuard inferred domain: {domain_guard.get('inferred_domain', '-')}",
        f"- DomainGuard corrected items: {domain_guard.get('corrected_items_count', 0)}",
        f"- DomainGuard leakage risk score: {domain_guard.get('domain_leakage_risk_score', 0)}",
        "",
        "Paperability Score:",
        f"- Total score: {paperability.get('total_score', '-')}",
        f"- Level: {paperability.get('level', '-')} / {paperability.get('level_tr', '-')}",
        "- Key reasons:",
        *paperability_reasons,
        f"- Recommended next action: {paperability.get('recommended_next_action', '-')}",
        "- Note: Bu skor karar destek amaçlı tahmini bir değerlendirmedir; yayın garantisi anlamına gelmez.",
        "",
        "Önerilen araştırma alanları:",
        *suggestion_lines,
        "",
        "Uyarılar ve hatalar:",
        *issue_lines,
        "",
        f"Export klasörü: {export_dir.resolve()}",
        "",
    ])


def build_sidebar_config() -> dict:
    is_demo = demo_mode_enabled()
    query_max_chars = 160 if is_demo else None
    with st.sidebar:
        st.header("Veri Kaynağı ve Analiz Ayarları")
        selected_domain = st.selectbox(
            "Araştırma Alanı",
            ACTIVE_RESEARCH_DOMAINS,
            index=0,
            key="selected_research_domain",
        )
        st.caption(
            "ResearchMind AI şu anda sağlık/biyomedikal ve mühendislik/uygulamalı teknolojiler "
            "alanlarında optimize edilmiştir. Diğer alanlar yakında aktif olacaktır."
        )
        source_label = st.selectbox(
            "Veri kaynağı",
            list(DATA_SOURCE_LABELS.values()),
            index=0,
            key="sidebar_data_source",
        )
        data_source = next(
            key for key, value in DATA_SOURCE_LABELS.items() if value == source_label
        )
        sync_pending_research_topic(data_source)
        transfer_notice = st.session_state.pop("topic_transfer_notice", "")
        if transfer_notice:
            st.success(transfer_notice)

        config = {
            "data_source": data_source,
            "data_source_label": source_label,
            "query": DEFAULT_LOCAL_QUERY,
            "csv_path": "biomedical_research_abstracts_2024_2026.csv",
            "row_limit": 20000,
            "openalex_api_key": "",
            "openalex_max_results": DEMO_LIVE_RESULT_LIMIT if is_demo else DEFAULT_LIVE_RESULT_LIMIT,
            "pubmed_max_results": DEMO_LIVE_RESULT_LIMIT if is_demo else DEFAULT_LIVE_RESULT_LIMIT,
            "pubmed_email": "",
            "pubmed_api_key": "",
            "years_back": 5,
            "selected_domain": selected_domain,
        }

        if data_source == "Local CSV":
            config["csv_path"] = st.text_input(
                "Veri dosyası yolu",
                value=config["csv_path"],
                key="local_csv_path",
            )
            config["row_limit"] = st.number_input(
                "Veri setinden kullanılacak kayıt sayısı",
                min_value=0,
                value=config["row_limit"],
                step=5000,
                help="0 seçilirse veri dosyasındaki tüm kayıtlar kullanılır.",
                key="local_row_limit",
            )
            config["query"] = st.text_input(
                "Araştırma konusu",
                value=DEFAULT_LOCAL_QUERY,
                help=QUERY_HELP,
                max_chars=query_max_chars,
                key="local_analysis_query",
            )
            render_query_help()

        elif data_source == "OpenAlex Live":
            config["query"] = st.text_input(
                "Araştırma konusu",
                value=DEFAULT_LIVE_QUERY,
                help=QUERY_HELP,
                max_chars=query_max_chars,
                key="openalex_query",
            )
            render_query_help()
            config["years_back"] = st.number_input(
                "Son kaç yıl analiz edilsin?",
                min_value=1,
                max_value=5,
                value=5,
                step=1,
                help="Canlı veri kaynaklarında son 1-5 yıl arasındaki güncel yayınlar analiz edilir.",
                key="openalex_years_back",
            )
            if not is_demo or is_admin():
                config["openalex_api_key"] = st.text_input(
                    "OpenAlex e-posta veya API bilgisi",
                    value="",
                    type="password",
                    help="Opsiyonel. Yoğun kullanımda önerilir.",
                    key="openalex_api_key",
                )

        elif data_source == "PubMed Live":
            config["query"] = st.text_input(
                "Araştırma konusu",
                value=DEFAULT_LIVE_QUERY,
                help=QUERY_HELP,
                max_chars=query_max_chars,
                key="pubmed_query",
            )
            render_query_help()
            config["years_back"] = st.number_input(
                "Son kaç yıl analiz edilsin?",
                min_value=1,
                max_value=5,
                value=5,
                step=1,
                help="Canlı veri kaynaklarında son 1-5 yıl arasındaki güncel yayınlar analiz edilir.",
                key="pubmed_years_back",
            )
            if not is_demo or is_admin():
                config["pubmed_email"] = st.text_input(
                    "NCBI e-posta adresi",
                    value="",
                    help="Opsiyonel, ancak NCBI tarafından önerilir.",
                    key="pubmed_email",
                )
                config["pubmed_api_key"] = st.text_input(
                    "PubMed API anahtarı",
                    value="",
                    type="password",
                    help="Opsiyonel. Girilirse daha kararlı erişim sağlayabilir.",
                    key="pubmed_api_key",
                )
                if not (config["pubmed_email"].strip() or get_pubmed_config().email):
                    st.warning(
                        "NCBI e-posta adresi eklenmemiş. Sistem çalışabilir; ancak yoğun kullanımda PubMed erişimi için e-posta önerilir."
                    )

        else:
            config["query"] = st.text_input(
                "Araştırma konusu",
                value=DEFAULT_LIVE_QUERY,
                help=QUERY_HELP,
                max_chars=query_max_chars,
                key="hybrid_query",
            )
            render_query_help()
            config["years_back"] = st.number_input(
                "Son kaç yıl analiz edilsin?",
                min_value=1,
                max_value=5,
                value=5,
                step=1,
                help="Canlı veri kaynaklarında son 1-5 yıl arasındaki güncel yayınlar analiz edilir.",
                key="hybrid_years_back",
            )
            if not is_demo or is_admin():
                config["openalex_api_key"] = st.text_input(
                    "OpenAlex e-posta veya API bilgisi",
                    value="",
                    type="password",
                    help="Opsiyonel. Yoğun kullanımda önerilir.",
                    key="hybrid_openalex_api_key",
                )
                config["pubmed_email"] = st.text_input(
                    "NCBI e-posta adresi",
                    value="",
                    help="Opsiyonel, ancak NCBI tarafından önerilir.",
                    key="hybrid_pubmed_email",
                )
                config["pubmed_api_key"] = st.text_input(
                    "PubMed API anahtarı",
                    value="",
                    type="password",
                    help="Opsiyonel. Girilirse daha kararlı erişim sağlayabilir.",
                    key="hybrid_pubmed_api_key",
                )
                if not (config["pubmed_email"].strip() or get_pubmed_config().email):
                    st.warning(
                        "NCBI e-posta adresi eklenmemiş. Sistem çalışabilir; ancak yoğun kullanımda PubMed erişimi için e-posta önerilir."
                    )

        if is_demo and len(str(config.get("query", ""))) > 160:
            config["query"] = str(config["query"])[:160]

        render_topic_suggester()

        st.divider()
        run_clicked = st.button(
            "Tüm Analizi Başlat",
            type="primary",
            use_container_width=True,
            key="run_full_analysis_button",
        )

        latest_export_path = st.session_state.get("latest_export_path", "")
        if latest_export_path:
            st.divider()
            if (not is_demo) or is_admin():
                st.caption("Son export klasörü")
                st.code(latest_export_path)
            _render_download_buttons(latest_export_path)

        render_config_warnings()
        render_admin_demo_management()

    return config | {"run_clicked": run_clicked}


def _render_download_buttons(export_path: str) -> None:
    export_dir = Path(export_path)
    normalized_path = export_dir / "normalized_dataset.csv"
    summary_path = export_dir / "summary_report.txt"
    executive_matches = sorted(export_dir.glob("ResearchMind_AI_Executive_Report_*.pdf"))
    executive_pdf_path = executive_matches[-1] if executive_matches else export_dir / "ResearchMind_AI_Executive_Report.pdf"
    text_pdf_path = export_dir / "ResearchMind_AI_Report.pdf"

    show_admin_exports = (not demo_mode_enabled()) or is_admin()

    if show_admin_exports and normalized_path.exists():
        st.download_button(
            "Normalize veri setini indir",
            data=normalized_path.read_bytes(),
            file_name="normalized_dataset.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_normalized_dataset",
        )

    if show_admin_exports and summary_path.exists():
        st.download_button(
            "Özet raporu indir",
            data=summary_path.read_bytes(),
            file_name="summary_report.txt",
            mime="text/plain",
            use_container_width=True,
            key="download_summary_report",
        )

    if executive_pdf_path.exists():
        st.download_button(
            label="Executive PDF İndir",
            data=executive_pdf_path.read_bytes(),
            file_name=executive_pdf_path.name,
            mime="application/pdf",
            use_container_width=True,
            key="download_executive_pdf_report_sidebar",
        )
    elif text_pdf_path.exists():
        st.download_button(
            label="Executive PDF İndir",
            data=text_pdf_path.read_bytes(),
            file_name="ResearchMind_AI_Executive_Report.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="download_executive_pdf_report_sidebar_fallback",
        )
    elif not text_pdf_path.exists():
        st.info("PDF raporu henüz oluşturulmadı.")


def render_results(results: dict) -> None:
    df = _as_dataframe(results.get("normalized_dataset"))
    diagnostics = results.get("diagnostics", {})
    pubmed_user_message = diagnostics.get("pubmed_user_message", "")

    for warning in results.get("warnings", []):
        st.warning(warning)

    for error in results.get("errors", []):
        if error == pubmed_user_message:
            continue
        st.error(error)

    _render_product_success(results)

    c1, c2, c3 = st.columns(3)
    c1.metric("Kayıt", f"{len(df):,}")
    c2.metric("Dergi", df["journal"].nunique() if "journal" in df.columns else "-")
    c3.metric("Araştırma Türü", df["research_type"].nunique() if "research_type" in df.columns else "-")

    _render_source_distribution(results)
    _render_pubmed_diagnostics(results)

    st.divider()
    _render_publication_trend(results)
    _render_topics(results)
    _render_distribution_tables(results)
    _render_query_gap(results)
    _render_semantic_results(results)
    _render_openalex_gap(results)
    _render_ai_suggestions(results)


def _render_source_distribution(results: dict) -> None:
    distribution = results.get("source_distribution", {})
    st.markdown("### Veri Kaynağı Dağılımı")
    s1, s2, s3 = st.columns(3)

    with s1:
        render_metric_card("PubMed", f"{distribution.get('PubMed', 0):,}", "Normalize edilen PubMed kaydı")
    with s2:
        render_metric_card("OpenAlex", f"{distribution.get('OpenAlex', 0):,}", "OpenAlex canlı veri kaydı")
    with s3:
        render_metric_card("Yerel", f"{distribution.get('Yerel', 0):,}", "Yerel veri seti kaydı")


def _render_pubmed_diagnostics(results: dict) -> None:
    diagnostics = results.get("diagnostics", {})

    if not diagnostics.get("pubmed_called"):
        return

    if diagnostics.get("pubmed_final_status") == "no_results":
        st.info(
            diagnostics.get(
                "pubmed_user_message",
                "PubMed araması bu spesifik konuda sonuç döndürmedi; OpenAlex ile analiz tamamlandı.",
            )
        )
        return

    if diagnostics.get("pubmed_error"):
        st.warning(
            diagnostics.get(
                "pubmed_user_message",
                "PubMed geçici olarak yanıt vermedi. OpenAlex sonuçlarıyla analiz tamamlandı.",
            )
        )
        if demo_mode_enabled() and not is_admin():
            return
        with st.expander("PubMed teknik detayını göster"):
            st.write(diagnostics["pubmed_error"])
            st.write(f"Final status: {diagnostics.get('pubmed_final_status') or '-'}")
            st.write(f"Original query: {diagnostics.get('pubmed_original_query') or '-'}")
            fallback_queries = diagnostics.get("pubmed_fallback_queries") or []
            st.write(f"Tried fallback queries: {', '.join(fallback_queries) if fallback_queries else '-'}")
        return

    st.info(
        "PubMed bağlantısı başarılı: "
        f"{diagnostics.get('pubmed_pmids_fetched', 0)} PMID çekildi, "
        f"{diagnostics.get('pubmed_normalized_records', 0)} kayıt normalize edildi."
        + (
            f" Kullanılan sorgu: {diagnostics.get('pubmed_successful_query')}."
            if diagnostics.get("pubmed_successful_query")
            else ""
        )
    )


def _render_publication_trend(results: dict) -> None:
    st.subheader("1) Yayın Trendi")
    trend = _as_dataframe(results.get("publication_trend"))

    if trend.empty:
        st.info("Yayın trendi üretilemedi.")
        return

    fig = px.line(
        trend,
        x="period",
        y="publication_count",
        markers=True,
        title="Yayın Trendi",
    )
    fig = style_plotly_chart(fig)
    st.plotly_chart(fig, use_container_width=True)


def _render_topics(results: dict) -> None:
    st.subheader("2) Öne Çıkan Biyomedikal Konular ve Anahtar Kelimeler")
    left, right = st.columns(2)

    with left:
        top_topics = _as_dataframe(results.get("top_topics"))

        if top_topics.empty:
            st.info("Konu verisi üretilemedi.")
        else:
            fig = px.bar(
                top_topics,
                x="topic_count",
                y="topic",
                orientation="h",
                title="Öne Çıkan Biyomedikal Konular",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            fig = style_plotly_chart(fig)
            st.plotly_chart(fig, use_container_width=True)

    with right:
        top_keywords = _as_dataframe(results.get("top_keywords"))

        if top_keywords.empty:
            st.info("Anahtar kelime verisi üretilemedi.")
        else:
            fig = px.bar(
                top_keywords,
                x="keyword_count",
                y="keyword",
                orientation="h",
                title="Öne Çıkan Anahtar Kelimeler / MeSH Terimleri",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            fig = style_plotly_chart(fig)
            st.plotly_chart(fig, use_container_width=True)


def _render_distribution_tables(results: dict) -> None:
    st.subheader("3) Dergi ve Araştırma Türü Analizi")
    a, b = st.columns(2)

    for label, key, container in [
        ("dergi", "journal_analysis", a),
        ("araştırma türü", "research_type_analysis", b),
    ]:
        with container:
            st.markdown(f"**Öne çıkan {label}**")
            out = _as_dataframe(results.get(key))
            out, message = clean_display_table(out, label)

            if out.empty:
                st.info(message)
            else:
                st.dataframe(out, use_container_width=True, hide_index=True)


def _render_query_gap(results: dict) -> None:
    st.subheader("4) Konu Trendi ve Research Gap Score")
    gap = results.get("gap_score") or {}
    qtrend = _as_dataframe(results.get("query_trend"))
    strategic_score = results.get("strategic_opportunity_score", gap.get("gap_score", "-"))
    opportunity_label, opportunity_level = strategic_level(strategic_score)
    big_badge, big_badge_level = opportunity_status(strategic_score)

    g1, g2, g3, g4 = st.columns(4)
    with g1:
        render_metric_card("Eşleşen kayıt", gap.get("total_records", 0), "Analize dahil olan ilgili yayın")
    with g2:
        render_metric_card("Büyüme oranı", gap.get("growth_rate", "-"), "Son dönem eğilim sinyali")
    with g3:
        render_metric_card("Strategic Opportunity Score", strategic_score, f"Ham Gap Score: {gap.get('gap_score', '-')}", big_badge, big_badge_level)
    with g4:
        render_metric_card("Research Opportunity Level", opportunity_label, "Skora göre stratejik seviye", big_badge, opportunity_level)

    insight = results.get("ai_research_insight") or generate_ai_insight(
        gap,
        _as_dataframe(results.get("top_topics")),
        _as_dataframe(results.get("top_keywords")),
        qtrend,
    )
    st.markdown("### AI Research Insight")
    st.markdown(
        f"""
        <div class="rm-insight">
            <div class="rm-insight-title">✦ Strategic Research Commentary</div>
            {insight}
        </div>
        """,
        unsafe_allow_html=True,
    )

    _render_research_strategy_engine(results)
    _render_domain_reasoning_summary(results)
    _render_paperability_score(results)

    interpretation = localize_text(gap.get("interpretation", ""))
    if interpretation:
        st.caption(interpretation)
    if "strict_matched_records" in gap:
        st.caption(
            f"Semantic matching aktif: eski strict eşleşme {gap.get('strict_matched_records', 0)}, "
            f"semantic eşleşme {gap.get('total_records', 0)} kayıt."
        )

    if qtrend.empty:
        st.info("Bu araştırma konusu için zamansal trend bulunamadı.")
        return

    fig = px.line(
        qtrend,
        x="period",
        y="publication_count",
        markers=True,
        title=f"Konu Trendi: {results.get('query', '')}",
    )
    fig = style_plotly_chart(fig)
    fig.update_layout(height=360, margin={"l": 18, "r": 18, "t": 48, "b": 28})
    st.plotly_chart(fig, use_container_width=True)
    trend_badge, trend_level = opportunity_status(strategic_score)
    st.markdown(
        f'<span class="rm-status rm-status-{trend_level}">{trend_badge}</span>',
        unsafe_allow_html=True,
    )


def _render_semantic_results(results: dict) -> None:
    st.subheader("5) Benzer Akademik Çalışmalar")
    st.caption("Girilen araştırma konusuna semantik olarak en yakın çalışmalar listelenir.")
    semantic = _as_dataframe(results.get("semantic_search_results"))

    if semantic.empty:
        st.info("Semantik arama sonucu üretilemedi.")
        return

    display = semantic.drop(columns=["abstract"], errors="ignore").copy()
    if "similarity_score" in display.columns:
        display["Benzerlik"] = (
            pd.to_numeric(display["similarity_score"], errors="coerce")
            .fillna(0)
            .map(lambda value: f"{value * 100:.1f}%")
        )
        display = display.drop(columns=["similarity_score"], errors="ignore")

    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("Özetleri göster"):
        for _, row in semantic.iterrows():
            st.markdown(f"**{row.get('title', 'Untitled')}**")
            st.caption(
                f"{row.get('journal', '')} | {row.get('pub_year', '')} | {row.get('country', '')}"
            )
            st.write(row.get("abstract", ""))
            st.divider()


def _render_research_strategy_engine(results: dict) -> None:
    strategy = results.get("research_strategy") or {}
    if not strategy:
        return

    st.markdown("### Research Strategy Engine")
    a, b = st.columns(2)
    c, d = st.columns(2)

    blocks = [
        ("Suggested Research Direction", strategy.get("direction", "-"), a),
        ("Recommended Methodology", strategy.get("methodology", "-"), b),
        ("Suggested Dataset / Evidence Focus", strategy.get("evidence", "-"), c),
        ("Differentiation Strategy", strategy.get("differentiation", "-"), d),
    ]

    for title, body, container in blocks:
        with container:
            st.markdown(
                f"""
                <div class="rm-card">
                    <div class="rm-card-label">{title}</div>
                    <div class="rm-card-note">{body}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_domain_reasoning_summary(results: dict) -> None:
    reasoning = results.get("domain_reasoning") or {}
    if not reasoning:
        return

    consistency = reasoning.get("domain_consistency", "-")
    risk = reasoning.get("semantic_leakage_risk", "-")
    level = "high" if consistency == "High" else "mid" if consistency == "Medium" else "low"
    risk_level = "low" if risk == "High" else "mid" if risk == "Medium" else "high"

    st.markdown("### Domain Reasoning Summary")
    cols = st.columns(5)
    values = [
        ("Primary domain", reasoning.get("clinical_domain", "-"), "Clinical family"),
        ("Primary modality", reasoning.get("primary_modality", "-"), "Evidence type"),
        ("Dominant methodology", reasoning.get("primary_method", "-"), "Method family"),
        ("Domain consistency", consistency, f"Score: {reasoning.get('domain_consistency_score', '-')}"),
        ("Semantic leakage risk", risk, f"Filtered: {reasoning.get('leakage_filtered_count', 0)}"),
    ]

    for index, (label, value, note) in enumerate(values):
        with cols[index]:
            badge = value if label in {"Domain consistency", "Semantic leakage risk"} else ""
            badge_level = level if label == "Domain consistency" else risk_level
            render_metric_card(label, value, note, badge if label in {"Domain consistency", "Semantic leakage risk"} else "", badge_level)

    if reasoning.get("leakage_filtered_count", 0):
        st.warning("Cross-domain semantic leakage detected and filtered.")


def _render_paperability_score(results: dict) -> None:
    paperability = results.get("paperability_score") or {}
    if not paperability:
        return

    score = parse_numeric(paperability.get("total_score"))
    level_en = paperability.get("level", "-")
    level_tr = paperability.get("level_tr", "-")
    level_key = paperability.get("level_key", "mid")
    reasons = paperability.get("reasons", [])[:5]
    next_action = paperability.get("recommended_next_action", "-")

    st.markdown("### Paperability Score")
    st.caption("SCI Publication Potential")

    left, right = st.columns([1, 2])
    with left:
        render_metric_card(
            "Paperability Score",
            f"{score:.1f}",
            "Estimated SCI publication potential",
            level_en,
            level_key,
        )
        st.progress(min(100, max(0, int(round(score)))) / 100)

    with right:
        reason_items = "".join(f"<li>{reason}</li>" for reason in reasons)
        st.markdown(
            f"""
            <div class="rm-insight">
                <div class="rm-insight-title">SCI Publication Potential: {level_tr}</div>
                <ul>{reason_items}</ul>
                <div class="rm-card-label">Recommended next action</div>
                <div>{next_action}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption("Bu skor karar destek amaçlı tahmini bir değerlendirmedir; yayın garantisi anlamına gelmez.")


def _render_openalex_gap(results: dict) -> None:
    openalex_gap = results.get("openalex_gap")

    if not openalex_gap:
        return

    st.subheader("6) OpenAlex Canlı Gap Analizi")
    l1, l2, l3 = st.columns(3)
    l1.metric("OpenAlex’te bulunan yayın", openalex_gap.get("total_records", 0))
    l2.metric("OpenAlex Büyüme Oranı", openalex_gap.get("growth_rate", "-"))
    l3.metric("OpenAlex Research Gap Score", openalex_gap.get("gap_score", "-"))
    st.write(localize_text(openalex_gap.get("interpretation", "")))

    trend = _as_dataframe(openalex_gap.get("trend"))
    if not trend.empty:
        st.dataframe(trend, use_container_width=True, hide_index=True)


def _render_product_success(results: dict) -> None:
    st.success("✅ Analiz başarıyla tamamlandı. Sonuçlar otomatik olarak kaydedildi.")
    if (not demo_mode_enabled()) or is_admin():
        with st.expander("Export klasörünü göster"):
            st.code(results.get("export_path", "-"))

    export_path = results.get("export_path", "")
    export_dir = Path(export_path) if export_path else Path()
    executive_matches = sorted(export_dir.glob("ResearchMind_AI_Executive_Report_*.pdf")) if export_path else []
    executive_pdf_path = executive_matches[-1] if executive_matches else export_dir / "ResearchMind_AI_Executive_Report.pdf"
    text_pdf_path = Path(export_path) / "ResearchMind_AI_Report.pdf" if export_path else Path()
    if export_path and executive_pdf_path.exists():
        st.download_button(
            label="Executive PDF İndir",
            data=executive_pdf_path.read_bytes(),
            file_name=executive_pdf_path.name,
            mime="application/pdf",
            use_container_width=True,
            key="download_executive_pdf_report_main",
        )
    elif export_path and text_pdf_path.exists():
        st.download_button(
            label="Executive PDF İndir",
            data=text_pdf_path.read_bytes(),
            file_name="ResearchMind_AI_Executive_Report.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="download_executive_pdf_report_main_fallback",
        )
    else:
        st.info("PDF raporu henüz oluşturulmadı.")


def _render_ai_suggestions(results: dict) -> None:
    st.subheader("7) Araştırma Konusu Önerileri")
    suggestions = _as_dataframe(results.get("ai_topic_suggestions"))

    if suggestions.empty:
        st.info("Araştırma konusu önerisi üretilemedi.")
        return

    st.markdown("### Öne Çıkan 3 Araştırma Fırsatı")
    top_three = suggestions.head(3)
    columns = st.columns(3)

    for idx, (_, row) in enumerate(top_three.iterrows()):
        with columns[idx]:
            title = row.get("suggested_research_topic", row.get("suggested_topic", "Araştırma önerisi"))
            score = row.get("gap_score", "-")
            growth = row.get("growth_rate", "-")
            recommendation = localize_text(row.get("recommendation", ""))
            trend_status, level = opportunity_trend_status(score, growth)
            st.markdown(
                f"""
                <div class="rm-opportunity">
                    <h4>{title}</h4>
                    <span class="rm-status rm-status-{level}">{trend_status}</span>
                    <div class="rm-muted" style="margin-top: 0.85rem;">Gap Score</div>
                    <div class="rm-card-value">{score}</div>
                    <div class="rm-card-note">Trend durumu: {growth}</div>
                    <p class="rm-card-note">{recommendation or "Bu fırsat detaylı tabloda incelenebilir."}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    display = suggestions.copy()
    if "recommendation" in display.columns:
        display["recommendation"] = display["recommendation"].map(localize_text)

    st.markdown("### Detaylı Öneri Tablosu")
    st.dataframe(display, use_container_width=True, hide_index=True)


inject_product_styles()

if not render_demo_registration_gate():
    st.stop()

sidebar_config = build_sidebar_config()

if sidebar_config["run_clicked"]:
    is_demo = demo_mode_enabled()
    demo_email = st.session_state.get("demo_user_email", "")
    domain_ok, domain_message, domain_debug = validate_domain_query(
        sidebar_config.get("query", ""),
        sidebar_config.get("selected_domain", current_selected_domain()),
    )
    st.session_state["domain_guard_debug"] = domain_debug

    if not domain_ok:
        st.warning(domain_message)
    elif is_demo and not st.session_state.get("demo_user_registered"):
        if domain_message:
            st.warning(domain_message)
        st.error("Demo analizi başlatmak için önce kayıt formunu doldurmanız gerekir.")
    elif is_demo and not is_admin() and demo_user_used_today(demo_email):
        if domain_message:
            st.warning(domain_message)
        st.warning("Bugünkü demo analiz hakkınız kullanılmıştır. İlginiz için teşekkür ederiz.")
    else:
        if domain_message:
            st.warning(domain_message)
        cached_results = load_demo_cache(sidebar_config) if is_demo else None

        if cached_results:
            log_demo_usage(demo_email, sidebar_config, "cached", cached_results.get("export_path", ""))
            st.session_state["analysis_results"] = cached_results
            st.session_state["latest_export_path"] = cached_results.get("export_path", "")
            st.session_state["domain_guard_debug"] = cached_results.get("domain_guard", st.session_state.get("domain_guard_debug", {}))
        else:
            with st.spinner("Tüm analiz çalıştırılıyor ve sonuçlar kaydediliyor..."):
                try:
                    analysis_results = run_full_analysis(sidebar_config)
                except Exception as exc:
                    analysis_results = {
                        "data_source": sidebar_config.get("data_source", "-"),
                        "data_source_label": sidebar_config.get("data_source_label", "-"),
                        "query": sidebar_config.get("query", "-"),
                        "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "normalized_dataset": pd.DataFrame(),
                        "warnings": [],
                        "errors": [f"Tüm analiz başarısız oldu: {exc}"],
                        "export_path": "",
                    }

                st.session_state["analysis_results"] = analysis_results
                st.session_state["latest_export_path"] = analysis_results.get("export_path", "")
                st.session_state["domain_guard_debug"] = analysis_results.get("domain_guard", st.session_state.get("domain_guard_debug", {}))

                if is_demo:
                    df = _as_dataframe(analysis_results.get("normalized_dataset"))
                    success = bool(analysis_results.get("export_path")) and not df.empty
                    if success:
                        save_demo_cache(sidebar_config, analysis_results)
                        log_demo_usage(demo_email, sidebar_config, "success", analysis_results.get("export_path", ""))
                    else:
                        log_demo_usage(demo_email, sidebar_config, "failed", analysis_results.get("export_path", ""))

results = st.session_state.get("analysis_results")
render_hero(results)

if results:
    render_results(results)
else:
    st.info("Araştırma konusunu gir, analiz dönemini seç ve tek tıkla trend, fırsat ve Research Gap Score sonuçlarını üret.")
