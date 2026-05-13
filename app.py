from __future__ import annotations

import base64
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
LOGO_PATH = Path(__file__).parent / "Researchmind logo.png"


def logo_data_uri() -> str:
    if not LOGO_PATH.exists():
        return ""
    try:
        encoded = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    except Exception:
        return ""
    return f"data:image/png;base64,{encoded}"

DATA_SOURCE_LABELS = {
    "Local CSV": "Yerel Veri Seti",
    "OpenAlex Live": "OpenAlex Canlı Veri",
    "PubMed Live": "PubMed Canlı Veri",
    "Hybrid: OpenAlex + PubMed": "Hibrit Analiz",
}

QUERY_WIDGET_KEYS = {
    "Local CSV": "local_analysis_query",
    "OpenAlex Live": "openalex_query",
    "PubMed Live": "pubmed_query",
    "Hybrid: OpenAlex + PubMed": "hybrid_query",
}

HEALTHCARE_DOMAIN = "Healthcare & Biomedical Sciences"
ENGINEERING_DOMAIN = "Engineering & Applied Technologies"
BIOMEDICAL_FIELD = "Biomedical Engineering"
SUPPORTED_FIELDS = [BIOMEDICAL_FIELD]
FIELD_TO_DOMAIN_FAMILY = {BIOMEDICAL_FIELD: "biomedical_engineering"}
LEGACY_DOMAIN_TO_FIELD = {
    ENGINEERING_DOMAIN: BIOMEDICAL_FIELD,
    HEALTHCARE_DOMAIN: BIOMEDICAL_FIELD,
    "Electrical & Electronics Engineering": BIOMEDICAL_FIELD,
    "AI-Compatible Medicine": BIOMEDICAL_FIELD,
    "Nursing": BIOMEDICAL_FIELD,
    "Midwifery": BIOMEDICAL_FIELD,
}
ACTIVE_RESEARCH_DOMAINS = SUPPORTED_FIELDS
DOMAIN_SUPPORT_WARNING = (
    "Bu demo sürüm yalnızca Biyomedikal Mühendisliği alanı için optimize edilmiştir. "
    "Lütfen konuyu biyosensörler, tıbbi cihazlar, biyomedikal sinyaller, "
    "giyilebilir sağlık sistemleri veya tıbbi görüntüleme bağlamında yeniden yazın."
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
    "clinical", "patient", "treatment", "healthcare", "clinical decision support",
    "hospital", "population", "medical", "clinically meaningful endpoints",
}

ENGINEERING_LANGUAGE_REPLACEMENTS = {
    "clinical decision support": "engineering decision support",
    "clinically meaningful endpoints": "engineering performance metrics",
    "clinical endpoints": "engineering performance metrics",
    "patient population": "target infrastructure or engineering system",
    "patient cohort": "engineering test cohort",
    "external clinical evidence": "external engineering validation evidence",
    "clinical evidence": "engineering validation evidence",
    "clinical validation": "engineering validation",
    "clinical records": "engineering system logs",
    "patient": "engineering system",
    "patients": "engineering systems",
    "diagnosis": "fault detection",
    "diagnostic": "fault-detection",
    "treatment": "mitigation strategy",
    "healthcare": "engineering systems",
    "hospital": "field deployment environment",
    "disease": "failure mode",
    "medical": "engineering",
    "population": "target infrastructure or engineering system",
    "clinical": "engineering",
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

ENGINEERING_SUBDOMAIN_ONTOLOGY = {
    "civil_structural_engineering": {
        "subdomain_label": "Civil / Structural Engineering",
        "keywords": [
            "civil engineering", "structural engineering", "seismic", "earthquake", "bridge", "building",
            "concrete", "steel", "geotechnical", "soil", "foundation", "tunnel", "dam", "flood",
            "infrastructure", "construction", "structural health monitoring", "fragility curve",
            "earthquake damage", "seismic vulnerability",
        ],
        "preferred_objects": ["bridges", "buildings", "infrastructure systems", "concrete structures", "steel structures", "soil-structure systems", "dams", "tunnels"],
        "preferred_metrics": ["seismic vulnerability", "structural damage", "load capacity", "displacement", "fragility", "resilience", "reliability", "safety"],
    },
    "materials_engineering": {
        "subdomain_label": "Materials Engineering",
        "keywords": [
            "materials engineering", "composite materials", "coating", "corrosion", "fatigue",
            "microstructure", "alloy", "polymer", "ceramics", "nanomaterials", "additive manufacturing",
            "heat treatment", "mechanical strength", "surface modification",
        ],
        "preferred_objects": ["composite materials", "coatings", "alloys", "polymer composites", "nanomaterials", "additively manufactured parts"],
        "preferred_metrics": ["corrosion resistance", "fatigue life", "mechanical strength", "hardness", "wear resistance", "microstructure-property relationship", "coating durability"],
    },
    "mechanical_engineering": {
        "subdomain_label": "Mechanical Engineering",
        "keywords": [
            "mechanical engineering", "cfd", "heat transfer", "thermodynamics", "fluid mechanics",
            "vibration", "tribology", "finite element analysis", "fatigue", "hvac", "turbine",
            "compressor", "engine", "thermal optimization",
        ],
        "preferred_objects": ["mechanical systems", "turbines", "heat exchangers", "engines", "HVAC systems", "rotating machinery"],
        "preferred_metrics": ["thermal efficiency", "vibration response", "stress distribution", "fatigue life", "pressure drop", "energy efficiency"],
    },
    "electrical_electronics_engineering": {
        "subdomain_label": "Electrical / Electronics Engineering",
        "keywords": [
            "electrical engineering", "electronics", "power systems", "smart grid", "renewable energy",
            "power electronics", "signal processing", "control systems", "embedded systems",
            "microcontroller", "sensor networks", "iot", "antenna", "wireless communication",
        ],
        "preferred_objects": ["power grids", "embedded systems", "sensor networks", "antennas", "control systems", "electronic circuits"],
        "preferred_metrics": ["energy efficiency", "signal quality", "voltage stability", "reliability", "latency", "power loss"],
    },
    "computer_software_engineering": {
        "subdomain_label": "Computer / Software Engineering",
        "keywords": [
            "software engineering", "computer engineering", "cybersecurity", "blockchain", "cloud computing",
            "edge computing", "machine learning", "deep learning", "computer vision", "data mining",
            "database", "distributed systems", "federated learning",
        ],
        "preferred_objects": ["software systems", "cyber-physical systems", "distributed networks", "blockchain systems", "AI models", "databases"],
        "preferred_metrics": ["accuracy", "latency", "robustness", "security", "scalability", "privacy", "computational efficiency"],
    },
    "aerospace_uav_engineering": {
        "subdomain_label": "Aerospace / UAV Engineering",
        "keywords": [
            "aerospace", "aviation", "uav", "drone", "flight control", "aerodynamics",
            "propulsion", "avionics", "satellite", "trajectory optimization", "swarm", "autonomous flight",
            "suru haberlesmesi", "ucus kontrol",
        ],
        "preferred_objects": ["UAV systems", "aircraft structures", "satellite systems", "flight controllers", "drone swarms"],
        "preferred_metrics": ["flight stability", "trajectory accuracy", "aerodynamic efficiency", "mission reliability", "communication latency"],
    },
    "chemical_process_engineering": {
        "subdomain_label": "Chemical / Process Engineering",
        "keywords": [
            "chemical engineering", "catalyst", "reaction kinetics", "process optimization", "membrane",
            "bioreactor", "mass transfer", "separation process", "chemical process", "reactor design",
        ],
        "preferred_objects": ["chemical reactors", "catalysts", "membranes", "separation systems", "bioreactors", "process plants"],
        "preferred_metrics": ["conversion efficiency", "selectivity", "yield", "mass transfer rate", "process stability", "energy consumption"],
    },
    "industrial_systems_engineering": {
        "subdomain_label": "Industrial / Systems Engineering",
        "keywords": [
            "industrial engineering", "optimization", "supply chain", "production planning", "scheduling",
            "quality control", "lean manufacturing", "decision support", "simulation", "logistics", "operations research",
        ],
        "preferred_objects": ["production systems", "supply chains", "logistics networks", "manufacturing lines", "decision support systems"],
        "preferred_metrics": ["cost reduction", "throughput", "resource utilization", "waiting time", "productivity", "quality improvement"],
    },
    "energy_environmental_engineering": {
        "subdomain_label": "Energy / Environmental Engineering",
        "keywords": [
            "energy systems", "renewable energy", "solar", "wind turbine", "photovoltaic", "battery",
            "hydrogen", "environmental engineering", "wastewater", "air pollution", "carbon emissions",
            "sustainability", "life cycle assessment", "kestirimci bakim",
        ],
        "preferred_objects": ["solar panels", "wind turbines", "batteries", "hydrogen systems", "wastewater systems", "environmental monitoring systems"],
        "preferred_metrics": ["energy yield", "efficiency", "emissions reduction", "degradation", "sustainability", "environmental impact", "reliability"],
    },
    "biomedical_engineering": {
        "subdomain_label": "Biomedical Engineering",
        "keywords": [
            "biomedical engineering", "biosensor", "prosthesis", "rehabilitation robotics", "medical device",
            "wearable sensor", "ecg", "eeg", "mri", "ct", "biomedical signal", "medical imaging",
        ],
        "preferred_objects": ["biosensors", "medical devices", "wearable systems", "prosthetic systems", "biomedical signals"],
        "preferred_metrics": ["diagnostic accuracy", "signal quality", "device reliability", "usability", "safety"],
    },
    "general_engineering": {
        "subdomain_label": "General Engineering",
        "keywords": ["engineering", "optimization", "simulation", "monitoring", "reliability"],
        "preferred_objects": ["engineering systems", "sensor-enabled systems", "simulation models", "benchmark systems"],
        "preferred_metrics": ["reliability", "efficiency", "robustness", "performance", "scalability"],
    },
}

HEALTHCARE_SUBDOMAIN_ONTOLOGY = {
    "medicine_clinical": {
        "subdomain_label": "Clinical Medicine",
        "keywords": ["medicine", "clinical medicine", "internal medicine", "diagnosis", "treatment", "prognosis", "disease", "patient", "clinical decision support", "clinical risk prediction", "therapy", "clinical outcome", "cancer", "oncology"],
        "preferred_objects": ["patients", "clinical cohorts", "disease groups", "diagnostic workflows", "treatment pathways"],
        "preferred_metrics": ["diagnostic accuracy", "prognosis", "treatment response", "clinical outcome", "mortality risk", "readmission risk"],
    },
    "nursing": {
        "subdomain_label": "Nursing",
        "keywords": ["nursing", "nurse", "nursing care", "patient care", "care quality", "nursing workload", "burnout", "triage", "patient safety", "clinical nursing", "nursing education"],
        "preferred_objects": ["nursing care processes", "nurses", "hospital wards", "patient safety systems", "nursing education programs"],
        "preferred_metrics": ["care quality", "workload", "burnout risk", "patient safety", "triage accuracy", "nursing performance"],
    },
    "midwifery": {
        "subdomain_label": "Midwifery",
        "keywords": ["midwifery", "midwife", "maternal health", "pregnancy", "prenatal care", "antenatal care", "postpartum", "childbirth", "birth outcomes", "neonatal", "obstetric care"],
        "preferred_objects": ["pregnant women", "maternal health services", "prenatal care pathways", "childbirth processes", "neonatal outcomes"],
        "preferred_metrics": ["maternal risk", "birth outcomes", "prenatal care quality", "postpartum complications", "neonatal safety"],
    },
    "biomedical_engineering_health": {
        "subdomain_label": "Biomedical Engineering / Health Technologies",
        "keywords": ["biomedical engineering", "medical device", "biosensor", "wearable sensor", "prosthesis", "rehabilitation robotics", "biomedical signal", "ecg", "eeg", "mri", "ct", "medical imaging", "bioinstrumentation"],
        "preferred_objects": ["medical devices", "biosensors", "wearable health systems", "biomedical signals", "imaging systems", "prosthetic systems"],
        "preferred_metrics": ["device reliability", "signal quality", "diagnostic accuracy", "usability", "safety", "measurement precision"],
    },
    "public_health": {
        "subdomain_label": "Public Health",
        "keywords": ["public health", "epidemiology", "health policy", "population health", "vaccination", "disease surveillance", "health inequality", "environmental health", "community health", "outbreak"],
        "preferred_objects": ["populations", "communities", "surveillance systems", "public health programs", "epidemiological datasets"],
        "preferred_metrics": ["incidence", "prevalence", "risk factors", "intervention effectiveness", "health equity", "outbreak prediction"],
    },
    "dentistry": {
        "subdomain_label": "Dentistry",
        "keywords": ["dentistry", "dental", "oral health", "caries", "periodontal", "orthodontics", "implant", "maxillofacial", "dental imaging"],
        "preferred_objects": ["dental patients", "oral health datasets", "dental images", "periodontal assessments", "implant systems"],
        "preferred_metrics": ["caries detection", "periodontal risk", "treatment success", "implant stability", "oral health outcome"],
    },
    "pharmacy_pharmacology": {
        "subdomain_label": "Pharmacy / Pharmacology",
        "keywords": ["pharmacy", "pharmacology", "drug", "medication", "adverse drug reaction", "pharmacovigilance", "dosage", "drug interaction", "medication adherence"],
        "preferred_objects": ["medications", "prescriptions", "drug safety records", "pharmacovigilance databases", "treatment regimens"],
        "preferred_metrics": ["drug safety", "adverse reaction risk", "medication adherence", "dosage optimization", "therapeutic response"],
    },
    "nutrition_dietetics": {
        "subdomain_label": "Nutrition and Dietetics",
        "keywords": ["nutrition", "dietetics", "diet", "dietary intake", "obesity", "malnutrition", "food consumption", "nutritional assessment", "metabolic health"],
        "preferred_objects": ["dietary records", "nutrition programs", "metabolic health profiles", "patient diets"],
        "preferred_metrics": ["nutritional status", "obesity risk", "malnutrition risk", "dietary adherence", "metabolic outcome"],
    },
    "physiotherapy_rehabilitation": {
        "subdomain_label": "Physiotherapy and Rehabilitation",
        "keywords": ["physiotherapy", "physical therapy", "rehabilitation", "exercise therapy", "gait analysis", "motor recovery", "musculoskeletal", "stroke rehabilitation"],
        "preferred_objects": ["rehabilitation programs", "patients undergoing therapy", "gait signals", "exercise interventions", "musculoskeletal assessments"],
        "preferred_metrics": ["functional recovery", "mobility improvement", "pain reduction", "gait performance", "rehabilitation outcome"],
    },
    "mental_health_psychology": {
        "subdomain_label": "Mental Health / Psychology",
        "keywords": ["mental health", "psychology", "psychiatry", "depression", "anxiety", "stress", "burnout", "cognitive assessment", "behavioral health"],
        "preferred_objects": ["mental health assessments", "psychological scales", "behavioral datasets", "therapy programs"],
        "preferred_metrics": ["symptom severity", "stress level", "burnout risk", "treatment response", "cognitive performance"],
    },
    "general_healthcare": {
        "subdomain_label": "General Healthcare",
        "keywords": ["healthcare", "medical", "clinical", "digital health"],
        "preferred_objects": ["healthcare workflows", "health datasets", "care delivery systems", "decision support workflows"],
        "preferred_metrics": ["care quality", "risk prediction", "workflow efficiency", "safety", "outcome improvement"],
    },
}

TURKISH_ENGINEERING_MAP = {
    "malzeme mühendisliği": "materials engineering", "malzeme muhendisligi": "materials engineering",
    "kompozit malzemeler": "composite materials", "kaplama teknolojileri": "coating technologies",
    "korozyon direnci": "corrosion resistance", "yorulma analizi": "fatigue analysis",
    "ısıl işlem": "heat treatment", "isil islem": "heat treatment", "mikro yapı": "microstructure",
    "mikroyapı": "microstructure", "mekanik dayanım": "mechanical strength", "mekanik dayanim": "mechanical strength",
    "katkılı imalat": "additive manufacturing", "katkili imalat": "additive manufacturing",
    "polimer kompozit": "polymer composites", "inşaat mühendisliği": "civil engineering",
    "insaat muhendisligi": "civil engineering", "deprem": "earthquake", "köprü": "bridge",
    "kopru": "bridge", "beton": "concrete", "zemin": "soil", "makine mühendisliği": "mechanical engineering",
    "makine muhendisligi": "mechanical engineering", "akışkanlar": "fluid mechanics",
    "akiskanlar": "fluid mechanics", "ısı transferi": "heat transfer", "isi transferi": "heat transfer",
    "elektrik elektronik": "electrical electronics engineering", "güç sistemleri": "power systems",
    "guc sistemleri": "power systems", "gömülü sistemler": "embedded systems",
    "gomulu sistemler": "embedded systems", "havacılık": "aerospace", "havacilik": "aerospace",
    "iha": "uav", "İHA": "uav", "kimya mühendisliği": "chemical engineering",
    "kimya muhendisligi": "chemical engineering", "endüstri mühendisliği": "industrial engineering",
    "endustri muhendisligi": "industrial engineering", "enerji sistemleri": "energy systems",
    "çevre mühendisliği": "environmental engineering", "cevre muhendisligi": "environmental engineering",
    "biyomedikal mühendisliği": "biomedical engineering", "biyomedikal muhendisligi": "biomedical engineering",
    "rüzgar türbini": "wind turbine", "ruzgar turbini": "wind turbine", "kestirimci bakım": "predictive maintenance",
    "kestirimci bakim": "predictive maintenance", "sürü haberleşmesi": "swarm communication",
    "suru haberlesmesi": "swarm communication", "uçuş kontrol": "flight control", "ucus kontrol": "flight control",
}

TURKISH_HEALTHCARE_MAP = {
    "tıp": "medicine", "tip": "medicine", "klinik": "clinical", "tanı": "diagnosis",
    "tani": "diagnosis", "tedavi": "treatment", "prognoz": "prognosis", "hasta": "patient",
    "kanser": "cancer", "onkoloji": "oncology", "hemşirelik": "nursing", "hemsirelik": "nursing",
    "hemşire": "nurse", "hemsire": "nurse", "hasta bakımı": "patient care", "hasta bakimi": "patient care",
    "bakım kalitesi": "care quality", "bakim kalitesi": "care quality", "hasta güvenliği": "patient safety",
    "hasta guvenligi": "patient safety", "ebelik": "midwifery", "ebe": "midwife", "gebelik": "pregnancy",
    "doğum": "childbirth", "dogum": "childbirth", "anne sağlığı": "maternal health",
    "anne sagligi": "maternal health", "yenidoğan": "neonatal", "yenidogan": "neonatal",
    "biyomedikal": "biomedical engineering", "tıbbi cihaz": "medical device", "tibbi cihaz": "medical device",
    "biyosensör": "biosensor", "biyosensor": "biosensor", "giyilebilir sensör": "wearable sensor",
    "giyilebilir sensor": "wearable sensor", "tıbbi görüntüleme": "medical imaging",
    "tibbi goruntuleme": "medical imaging", "halk sağlığı": "public health", "halk sagligi": "public health",
    "epidemiyoloji": "epidemiology", "aşılama": "vaccination", "asilama": "vaccination",
    "ağız ve diş sağlığı": "oral health", "agiz ve dis sagligi": "oral health",
    "diş hekimliği": "dentistry", "dis hekimligi": "dentistry", "eczacılık": "pharmacy",
    "eczacilik": "pharmacy", "farmakoloji": "pharmacology", "ilaç etkileşimi": "drug interaction",
    "ilac etkilesimi": "drug interaction", "beslenme": "nutrition", "diyetetik": "dietetics",
    "fizyoterapi": "physiotherapy", "rehabilitasyon": "rehabilitation", "ruh sağlığı": "mental health",
    "ruh sagligi": "mental health", "psikoloji": "psychology", "depresyon": "depression", "anksiyete": "anxiety",
}


BIOMEDICAL_ENGINEERING_ONTOLOGY = {
    "keywords": [
        "biomedical engineering", "medical device", "medical devices", "biosensor", "biosensors",
        "wearable sensor", "wearable health system", "wearable health systems",
        "biomedical signal", "biomedical signal processing", "ecg", "eeg", "emg", "mri", "ct",
        "medical imaging", "bioinstrumentation", "prosthesis", "rehabilitation robotics",
        "medical device reliability", "health monitoring", "digital biomarker", "signal quality",
        "noise reduction", "artifact removal", "sensor fusion", "physiological signals",
        "physiological signal", "remote patient monitoring",
    ],
    "preferred_objects": [
        "biosensors", "wearable health systems", "biomedical signals", "medical devices",
        "medical imaging systems", "prosthetic systems", "rehabilitation robotics platforms",
        "physiological monitoring systems",
    ],
    "preferred_metrics": [
        "signal quality", "device reliability", "measurement precision", "diagnostic accuracy",
        "artifact reduction", "noise robustness", "usability", "safety",
        "biomedical signal classification performance",
    ],
    "validation_language": "Biyomedikal cihaz güvenilirliği, sinyal kalitesi, ölçüm hassasiyeti, güvenlik, kullanılabilirlik ve karşılaştırılabilir biyomedikal veri kümeleri üzerinden doğrulama önceliklendirilmelidir.",
}

TURKISH_BIOMEDICAL_MAP = {
    "biyomedikal mühendisliği": "biomedical engineering",
    "biyomedikal muhendisligi": "biomedical engineering",
    "biyomedikal": "biomedical engineering",
    "tıbbi cihaz": "medical device",
    "tibbi cihaz": "medical device",
    "tıbbi cihazlar": "medical devices",
    "tibbi cihazlar": "medical devices",
    "biyosensör": "biosensor",
    "biyosensor": "biosensor",
    "biyosensörler": "biosensors",
    "biyosensorler": "biosensors",
    "giyilebilir sensör": "wearable sensor",
    "giyilebilir sensor": "wearable sensor",
    "giyilebilir sağlık sistemi": "wearable health system",
    "giyilebilir saglik sistemi": "wearable health system",
    "giyilebilir sağlık sistemleri": "wearable health systems",
    "giyilebilir saglik sistemleri": "wearable health systems",
    "biyomedikal sinyal": "biomedical signal",
    "biyomedikal sinyal işleme": "biomedical signal processing",
    "biyomedikal sinyal isleme": "biomedical signal processing",
    "sinyal kalitesi": "signal quality",
    "gürültü azaltma": "noise reduction",
    "gurultu azaltma": "noise reduction",
    "artefakt giderme": "artifact removal",
    "sensör füzyonu": "sensor fusion",
    "sensor fuzyonu": "sensor fusion",
    "tıbbi görüntüleme": "medical imaging",
    "tibbi goruntuleme": "medical imaging",
    "protez": "prosthesis",
    "rehabilitasyon robotiği": "rehabilitation robotics",
    "rehabilitasyon robotigi": "rehabilitation robotics",
    "uzaktan hasta izleme": "remote patient monitoring",
    "fizyolojik sinyal": "physiological signal",
}

SUPPORTED_FIELD_ONTOLOGY = {BIOMEDICAL_FIELD: BIOMEDICAL_ENGINEERING_ONTOLOGY}
TURKISH_FIELD_KEYWORD_MAP = TURKISH_BIOMEDICAL_MAP

BIOMEDICAL_OUTSIDE_TERMS = set(BLOCKED_KEYWORDS) | {
    "civil engineering", "seismic", "earthquake", "bridge", "concrete", "structural engineering",
    "construction", "geotechnical", "materials engineering", "coating", "corrosion", "alloy",
    "polymer", "composite materials", "nursing", "nurse", "midwifery", "midwife",
    "care quality", "patient safety", "childbirth", "pregnancy", "neonatal", "tax", "fiscal",
    "economics", "public expenditure", "hemsirelik", "hemsire", "ebelik", "ebe", "gebelik",
    "dogum", "yenidogan", "hasta guvenligi", "bakim kalitesi", "malzeme muhendisligi",
    "kaplama", "korozyon", "insaat muhendisligi", "deprem", "kopru", "beton",
}
BIOMEDICAL_CLINICAL_DRIFT_TERMS = {
    "clinical cohorts", "patient groups", "treatment response", "prognostic performance",
    "prognosis", "mortality risk", "readmission risk", "clinical outcome prediction",
}
BIOMEDICAL_CONTEXT_TERMS = set(BIOMEDICAL_ENGINEERING_ONTOLOGY["keywords"])

BIOMEDICAL_TERMS = [
    "biomedical", "biomedical engineering", "biosensor", "biosensors",
    "medical device", "medical devices", "wearable sensor",
    "wearable health", "wearable biomedical", "biomedical signal",
    "biomedical signal processing", "ecg", "eeg", "emg",
    "medical imaging", "mri", "ct", "ultrasound",
    "prosthesis", "rehabilitation robotics", "health monitoring",
    "physiological signal", "signal quality", "artifact reduction", "noise reduction",
    "sensor fusion", "digital biomarker",
]

STRONG_BIOMEDICAL_OBJECTS = [
    "mr", "mri", "ct", "pet", "eeg", "ecg", "emg",
    "alzheimer", "dementia", "cancer", "tumor",
    "biosensor", "biosensors", "wearable", "medical imaging",
    "biomedical signal", "neurological", "radiology",
]

GENERAL_RELATED_TERMS = [
    "ai", "artificial intelligence", "machine learning", "deep learning",
    "healthcare", "health", "monitoring", "diagnosis",
    "digital health", "remote monitoring", "classification",
    "prediction", "risk prediction",
]

UNRELATED_BLOCK_TERMS = [
    "tax", "taxation", "fiscal", "economics", "inflation",
    "accounting", "audit", "public expenditure", "stock market",
    "cryptocurrency trading", "bridge", "concrete", "seismic",
    "earthquake", "civil engineering", "materials engineering",
    "coating", "corrosion", "football", "tourism", "hotel",
    "restaurant", "real estate",
]

TURKISH_UNRELATED_MAP = {
    "vergi": "tax",
    "ekonomi": "economics",
    "enflasyon": "inflation",
    "köprü": "bridge",
    "kopru": "bridge",
    "beton": "concrete",
    "deprem": "earthquake",
    "futbol": "football",
    "turizm": "tourism",
    "otel": "hotel",
}

BIOMEDICAL_SOFT_GUIDANCE_MESSAGE = (
    "Bu konu biyomedikal mühendisliği ile kısmen ilişkili görünüyor. Daha güçlü sonuçlar için "
    "biyosensör, tıbbi cihaz, biyomedikal sinyal, giyilebilir sağlık sistemi, ECG/EEG veya "
    "tıbbi görüntüleme gibi daha spesifik terimler eklemeniz önerilir."
)

BIOMEDICAL_HARD_BLOCK_MESSAGE = (
    "Bu demo sürüm yalnızca Biyomedikal Mühendisliği alanı için optimize edilmiştir. Girilen bazı "
    "terimler bu alanla eşleşmiyor. Lütfen konuyu biyosensörler, tıbbi cihazlar, biyomedikal "
    "sinyaller, giyilebilir sağlık sistemleri veya tıbbi görüntüleme bağlamında yeniden yazın."
)

BIOMEDICAL_KEYWORD_SUGGESTION = (
    "Örnek biyomedikal anahtar kelimeler:\n"
    "- ECG signal processing\n"
    "- wearable biosensors\n"
    "- medical imaging AI\n"
    "- biomedical signal analysis"
)

BIOMEDICAL_MIXED_TERM_WARNING = (
    "Bazı terimler biyomedikal mühendisliği ile güçlü eşleşmedi; analiz biyomedikal odaklı terimler üzerinden yürütülecek."
)



BIOMEDICAL_TOPIC_BANK = [
    {"title": 'Explainable AI for ECG Signal Quality Assessment in Wearable Biosensors', "tags": ['ecg', 'biosensor', 'wearable health systems', 'biomedical signal processing', 'explainable ai'], "rationale": 'Biyomedikal sinyal işleme ve sinyal kalitesi odağı korunarak önerildi.'},
    {"title": 'Noise-Robust ECG Analysis for Wearable Health Monitoring Systems', "tags": ['ecg', 'wearable health systems', 'noise reduction'], "rationale": 'Biyomedikal sinyal işleme ve sinyal kalitesi odağı korunarak önerildi.'},
    {"title": 'Deep Learning-Based Artifact Reduction in Biomedical Signal Processing', "tags": ['ct', 'biomedical signal processing', 'artifact reduction'], "rationale": 'Biyomedikal sinyal işleme ve sinyal kalitesi odağı korunarak önerildi.'},
    {"title": 'AI-Based Calibration Framework for Smart Biomedical Sensors', "tags": ['biomedical calibration'], "rationale": 'Biyomedikal kalibrasyon, hasta monitörü ve sinyal doğrulama odağı korunarak önerildi.'},
    {"title": 'Signal Quality Assessment in Bedside Patient Monitoring Devices', "tags": ['biomedical signal processing', 'bedside monitoring', 'patient monitoring'], "rationale": 'Biyomedikal kalibrasyon, hasta monitörü ve sinyal doğrulama odağı korunarak önerildi.'},
    {"title": 'Sensor Fusion-Based Reliability Assessment for Wearable Biosensor Systems', "tags": ['biosensor', 'wearable health systems', 'sensor fusion', 'biomedical reliability'], "rationale": 'Giyilebilir sistemler ve biyosensör bağlamı korunarak önerildi.'},
    {"title": 'Transformer-Based EEG Signal Classification for Alzheimer Detection', "tags": ['eeg', 'alzheimer', 'biomedical signal processing'], "rationale": 'Alzheimer, nörogörüntüleme veya nörolojik biyomedikal bağlam korunarak önerildi.'},
    {"title": 'Explainable Deep Learning for Early Alzheimer Diagnosis Using EEG Signals', "tags": ['eeg', 'alzheimer', 'biomedical signal processing', 'explainable ai'], "rationale": 'Alzheimer, nörogörüntüleme veya nörolojik biyomedikal bağlam korunarak önerildi.'},
    {"title": 'Multimodal MRI and Clinical Data Fusion for Alzheimer Screening', "tags": ['mri', 'alzheimer', 'multimodal fusion'], "rationale": 'Alzheimer, nörogörüntüleme veya nörolojik biyomedikal bağlam korunarak önerildi.'},
    {"title": 'Edge AI for Real-Time Physiological Signal Monitoring', "tags": ['biomedical signal processing', 'edge ai', 'physiological monitoring'], "rationale": 'Biyomedikal sinyal işleme ve sinyal kalitesi odağı korunarak önerildi.'},
    {"title": 'TinyML-Based ECG Monitoring on Wearable Medical Devices', "tags": ['ecg', 'wearable health systems', 'tinyml', 'smart medical devices'], "rationale": 'Biyomedikal sinyal işleme ve sinyal kalitesi odağı korunarak önerildi.'},
    {"title": 'IoMT-Enabled Remote Patient Monitoring Using Biomedical Sensors', "tags": ['patient monitoring', 'iomt', 'remote healthcare'], "rationale": 'Biyomedikal kalibrasyon, hasta monitörü ve sinyal doğrulama odağı korunarak önerildi.'},
    {"title": 'Biomedical Signal Quality Index Estimation for Remote Monitoring Systems', "tags": ['biomedical signal processing', 'remote healthcare'], "rationale": 'Biyomedikal sinyal işleme ve sinyal kalitesi odağı korunarak önerildi.'},
    {"title": 'AI-Assisted Predictive Maintenance for Smart Medical Devices', "tags": ['predictive maintenance', 'smart medical devices'], "rationale": 'Akıllı tıbbi cihaz güvenilirliği ve doğrulama odağıyla önerildi.'},
    {"title": 'Federated Learning for Privacy-Preserving Biomedical Signal Analysis', "tags": ['biomedical signal processing', 'federated learning'], "rationale": 'Biyomedikal sinyal işleme ve sinyal kalitesi odağı korunarak önerildi.'},
    {"title": 'Digital Twin-Based Reliability Assessment of Biomedical Monitoring Systems', "tags": ['digital twins', 'biomedical reliability'], "rationale": 'Akıllı tıbbi cihaz güvenilirliği ve doğrulama odağıyla önerildi.'},
    {"title": 'AI-Based Noise Reduction in EEG Signal Processing', "tags": ['eeg', 'biomedical signal processing', 'noise reduction'], "rationale": 'Biyomedikal sinyal işleme ve sinyal kalitesi odağı korunarak önerildi.'},
    {"title": 'Wearable Biosensor Systems for Continuous Physiological Monitoring', "tags": ['biosensor', 'wearable health systems', 'physiological monitoring'], "rationale": 'Giyilebilir sistemler ve biyosensör bağlamı korunarak önerildi.'},
    {"title": 'Machine Learning for Biomedical Sensor Calibration Drift Detection', "tags": ['biomedical calibration'], "rationale": 'Biyomedikal kalibrasyon, hasta monitörü ve sinyal doğrulama odağı korunarak önerildi.'},
    {"title": 'Deep Neural Networks for Biomedical Image Quality Enhancement', "tags": ['medical imaging'], "rationale": 'Tıbbi görüntüleme ve biyomedikal doğrulama odağıyla önerildi.'},
    {"title": 'AI-Based Ultrasound Image Enhancement for Clinical Diagnostics', "tags": ['ultrasound', 'medical imaging'], "rationale": 'Tıbbi görüntüleme ve biyomedikal doğrulama odağıyla önerildi.'},
    {"title": 'Explainable MRI Classification Models for Neurodegenerative Diseases', "tags": ['mri', 'neuroimaging', 'explainable ai'], "rationale": 'Alzheimer, nörogörüntüleme veya nörolojik biyomedikal bağlam korunarak önerildi.'},
    {"title": 'Deep Learning-Based ECG Arrhythmia Detection in Wearable Systems', "tags": ['ecg', 'wearable health systems'], "rationale": 'Biyomedikal sinyal işleme ve sinyal kalitesi odağı korunarak önerildi.'},
    {"title": 'Real-Time Biomedical Signal Compression for Edge Healthcare Devices', "tags": ['biomedical signal processing', 'edge ai'], "rationale": 'Biyomedikal sinyal işleme ve sinyal kalitesi odağı korunarak önerildi.'},
    {"title": 'AI-Powered Respiratory Signal Analysis for Smart Monitoring Systems', "tags": ['biomedical signal processing'], "rationale": 'Biyomedikal sinyal işleme ve sinyal kalitesi odağı korunarak önerildi.'},
    {"title": 'Multimodal Physiological Signal Fusion for Smart Healthcare Systems', "tags": ['biomedical signal processing', 'physiological monitoring', 'multimodal fusion'], "rationale": 'Biyomedikal sinyal işleme ve sinyal kalitesi odağı korunarak önerildi.'},
    {"title": 'Biomedical Device Fault Detection Using Deep Learning', "tags": ['smart medical devices'], "rationale": 'Akıllı tıbbi cihaz güvenilirliği ve doğrulama odağıyla önerildi.'},
    {"title": 'Explainable AI for Smart ICU Monitoring Systems', "tags": ['explainable ai'], "rationale": 'Girilen anahtar kelimelerle uyumlu biyomedikal mühendisliği başlığı önerildi.'},
    {"title": 'Real-Time ECG Artifact Detection for Wearable Healthcare Devices', "tags": ['ecg', 'ct', 'wearable health systems', 'artifact reduction'], "rationale": 'Biyomedikal sinyal işleme ve sinyal kalitesi odağı korunarak önerildi.'},
    {"title": 'AI-Based Biomedical Signal Classification for Clinical Decision Support', "tags": ['biomedical signal processing'], "rationale": 'Biyomedikal sinyal işleme ve sinyal kalitesi odağı korunarak önerildi.'},
    {"title": 'Deep Learning for EEG-Based Cognitive Decline Detection', "tags": ['eeg'], "rationale": 'Biyomedikal sinyal işleme ve sinyal kalitesi odağı korunarak önerildi.'},
    {"title": 'Smart Biosensor Networks for Continuous Patient Monitoring', "tags": ['biosensor', 'patient monitoring'], "rationale": 'Biyomedikal kalibrasyon, hasta monitörü ve sinyal doğrulama odağı korunarak önerildi.'},
    {"title": 'AI-Assisted Medical Device Calibration Validation Framework', "tags": ['biomedical calibration', 'smart medical devices'], "rationale": 'Biyomedikal kalibrasyon, hasta monitörü ve sinyal doğrulama odağı korunarak önerildi.'},
    {"title": 'Machine Learning-Based Biomedical Signal Denoising Techniques', "tags": ['biomedical signal processing'], "rationale": 'Biyomedikal sinyal işleme ve sinyal kalitesi odağı korunarak önerildi.'},
    {"title": 'Transformer Models for EEG-Based Emotion Recognition', "tags": ['eeg'], "rationale": 'Biyomedikal sinyal işleme ve sinyal kalitesi odağı korunarak önerildi.'},
    {"title": 'Explainable AI for CT Image Classification in Clinical Diagnostics', "tags": ['ct', 'explainable ai', 'medical imaging'], "rationale": 'Tıbbi görüntüleme ve biyomedikal doğrulama odağıyla önerildi.'},
    {"title": 'Edge Computing for Real-Time Biomedical Monitoring Applications', "tags": ['edge ai'], "rationale": 'Girilen anahtar kelimelerle uyumlu biyomedikal mühendisliği başlığı önerildi.'},
    {"title": 'Wearable EEG Systems for Continuous Neurological Assessment', "tags": ['eeg', 'neuroimaging', 'wearable health systems'], "rationale": 'Alzheimer, nörogörüntüleme veya nörolojik biyomedikal bağlam korunarak önerildi.'},
    {"title": 'Biomedical Sensor Reliability Modeling Using Artificial Intelligence', "tags": ['biomedical reliability'], "rationale": 'Akıllı tıbbi cihaz güvenilirliği ve doğrulama odağıyla önerildi.'},
    {"title": 'Deep Learning-Assisted Biomedical Image Segmentation for MRI Analysis', "tags": ['mri', 'medical imaging'], "rationale": 'Tıbbi görüntüleme ve biyomedikal doğrulama odağıyla önerildi.'},
    {"title": 'AI-Based Sleep Apnea Detection Using Physiological Signals', "tags": ['biomedical signal processing', 'physiological monitoring'], "rationale": 'Biyomedikal sinyal işleme ve sinyal kalitesi odağı korunarak önerildi.'},
    {"title": 'ECG Signal Compression for Remote Cardiac Monitoring', "tags": ['ecg', 'biomedical signal processing', 'remote healthcare'], "rationale": 'Biyomedikal sinyal işleme ve sinyal kalitesi odağı korunarak önerildi.'},
    {"title": 'Wearable Sensor Fusion for Athlete Physiological Monitoring', "tags": ['wearable health systems', 'sensor fusion', 'physiological monitoring'], "rationale": 'Giyilebilir sistemler ve biyosensör bağlamı korunarak önerildi.'},
    {"title": 'TinyML Frameworks for Low-Power Biomedical Devices', "tags": ['tinyml', 'smart medical devices'], "rationale": 'Akıllı tıbbi cihaz güvenilirliği ve doğrulama odağıyla önerildi.'},
    {"title": 'Explainable AI in Smart Cardiac Monitoring Systems', "tags": ['explainable ai'], "rationale": 'Girilen anahtar kelimelerle uyumlu biyomedikal mühendisliği başlığı önerildi.'},
    {"title": 'AI-Assisted Blood Pressure Estimation Using Wearable Sensors', "tags": ['wearable health systems'], "rationale": 'Giyilebilir sistemler ve biyosensör bağlamı korunarak önerildi.'},
    {"title": 'Federated Learning for Distributed EEG Signal Analysis', "tags": ['eeg', 'biomedical signal processing', 'federated learning'], "rationale": 'Biyomedikal sinyal işleme ve sinyal kalitesi odağı korunarak önerildi.'},
    {"title": 'Biomedical Signal Enhancement Using Generative AI', "tags": ['biomedical signal processing'], "rationale": 'Biyomedikal sinyal işleme ve sinyal kalitesi odağı korunarak önerildi.'},
    {"title": 'Deep Learning-Based Motion Artifact Removal in ECG Signals', "tags": ['ecg', 'ct', 'biomedical signal processing', 'artifact reduction'], "rationale": 'Biyomedikal sinyal işleme ve sinyal kalitesi odağı korunarak önerildi.'},
    {"title": 'MRI-Based Brain Tissue Classification Using Explainable AI', "tags": ['mri', 'explainable ai'], "rationale": 'Girilen anahtar kelimelerle uyumlu biyomedikal mühendisliği başlığı önerildi.'},
    {"title": 'AI-Based Fall Detection Systems for Elderly Healthcare', "tags": ['biomedical engineering'], "rationale": 'Girilen anahtar kelimelerle uyumlu biyomedikal mühendisliği başlığı önerildi.'},
    {"title": 'Smart Rehabilitation Monitoring with Wearable Sensors', "tags": ['wearable health systems', 'rehabilitation systems'], "rationale": 'Giyilebilir sistemler ve biyosensör bağlamı korunarak önerildi.'},
    {"title": 'AI-Driven Biomedical Image Super-Resolution Techniques', "tags": ['medical imaging'], "rationale": 'Tıbbi görüntüleme ve biyomedikal doğrulama odağıyla önerildi.'},
    {"title": 'Deep Learning for Portable Ultrasound Image Analysis', "tags": ['ultrasound', 'medical imaging'], "rationale": 'Tıbbi görüntüleme ve biyomedikal doğrulama odağıyla önerildi.'},
    {"title": 'Real-Time Oxygen Saturation Monitoring with Smart Biosensors', "tags": ['biosensor'], "rationale": 'Giyilebilir sistemler ve biyosensör bağlamı korunarak önerildi.'},
    {"title": 'AI-Based Arrhythmia Prediction Using ECG Time-Series Data', "tags": ['ecg'], "rationale": 'Biyomedikal sinyal işleme ve sinyal kalitesi odağı korunarak önerildi.'},
    {"title": 'Wearable Health Monitoring with Edge AI Architectures', "tags": ['wearable health systems', 'edge ai'], "rationale": 'Giyilebilir sistemler ve biyosensör bağlamı korunarak önerildi.'},
    {"title": 'Biomedical Signal Segmentation Using Transformer Networks', "tags": ['biomedical signal processing'], "rationale": 'Biyomedikal sinyal işleme ve sinyal kalitesi odağı korunarak önerildi.'},
    {"title": 'Explainable Federated Learning for Clinical Signal Processing', "tags": ['biomedical signal processing', 'federated learning', 'explainable ai'], "rationale": 'Biyomedikal sinyal işleme ve sinyal kalitesi odağı korunarak önerildi.'},
    {"title": 'AI-Based Smart Stethoscope Signal Enhancement', "tags": ['biomedical signal processing'], "rationale": 'Biyomedikal sinyal işleme ve sinyal kalitesi odağı korunarak önerildi.'},
    {"title": 'Deep Learning for Automated EEG Artifact Detection', "tags": ['eeg', 'ct', 'artifact reduction'], "rationale": 'Biyomedikal sinyal işleme ve sinyal kalitesi odağı korunarak önerildi.'},
    {"title": 'Biomedical Device Reliability Prediction Using Digital Twins', "tags": ['digital twins', 'smart medical devices', 'biomedical reliability'], "rationale": 'Akıllı tıbbi cihaz güvenilirliği ve doğrulama odağıyla önerildi.'},
    {"title": 'Smart Sensor Calibration in Connected Medical Systems', "tags": ['biomedical calibration'], "rationale": 'Biyomedikal kalibrasyon, hasta monitörü ve sinyal doğrulama odağı korunarak önerildi.'},
    {"title": 'AI-Assisted Glucose Monitoring Using Wearable Biosensors', "tags": ['biosensor', 'wearable health systems'], "rationale": 'Giyilebilir sistemler ve biyosensör bağlamı korunarak önerildi.'},
    {"title": 'Predictive Modeling for Intensive Care Patient Monitoring', "tags": ['patient monitoring'], "rationale": 'Biyomedikal kalibrasyon, hasta monitörü ve sinyal doğrulama odağı korunarak önerildi.'},
    {"title": 'ECG-Based Stress Detection Using Machine Learning', "tags": ['ecg'], "rationale": 'Biyomedikal sinyal işleme ve sinyal kalitesi odağı korunarak önerildi.'},
    {"title": 'Biomedical Noise Filtering with Adaptive Deep Networks', "tags": ['noise reduction'], "rationale": 'Girilen anahtar kelimelerle uyumlu biyomedikal mühendisliği başlığı önerildi.'},
    {"title": 'MRI Reconstruction Using Deep Learning Architectures', "tags": ['mri'], "rationale": 'Girilen anahtar kelimelerle uyumlu biyomedikal mühendisliği başlığı önerildi.'},
    {"title": 'Explainable AI for Physiological Signal Classification', "tags": ['biomedical signal processing', 'explainable ai', 'physiological monitoring'], "rationale": 'Biyomedikal sinyal işleme ve sinyal kalitesi odağı korunarak önerildi.'},
    {"title": 'Wearable Multimodal Biosensors for Remote Healthcare', "tags": ['biosensor', 'wearable health systems', 'remote healthcare', 'multimodal fusion'], "rationale": 'Giyilebilir sistemler ve biyosensör bağlamı korunarak önerildi.'},
    {"title": 'AI-Based Neuroimaging Analysis for Dementia Detection', "tags": ['dementia', 'neuroimaging', 'medical imaging'], "rationale": 'Alzheimer, nörogörüntüleme veya nörolojik biyomedikal bağlam korunarak önerildi.'},
    {"title": 'Biomedical Sensor Fault Diagnosis in Smart Hospitals', "tags": ['biomedical engineering'], "rationale": 'Girilen anahtar kelimelerle uyumlu biyomedikal mühendisliği başlığı önerildi.'},
    {"title": 'TinyML-Assisted Biomedical Signal Monitoring', "tags": ['biomedical signal processing', 'tinyml'], "rationale": 'Biyomedikal sinyal işleme ve sinyal kalitesi odağı korunarak önerildi.'},
    {"title": 'Edge AI for Portable EEG Monitoring Devices', "tags": ['eeg', 'edge ai'], "rationale": 'Biyomedikal sinyal işleme ve sinyal kalitesi odağı korunarak önerildi.'},
    {"title": 'Deep Learning-Based Smart Prosthetic Control Systems', "tags": ['prosthetics'], "rationale": 'Girilen anahtar kelimelerle uyumlu biyomedikal mühendisliği başlığı önerildi.'},
    {"title": 'AI-Based Human Motion Analysis in Rehabilitation Engineering', "tags": ['rehabilitation systems'], "rationale": 'Girilen anahtar kelimelerle uyumlu biyomedikal mühendisliği başlığı önerildi.'},
    {"title": 'Explainable Biomedical Decision Support Systems', "tags": ['explainable ai'], "rationale": 'Girilen anahtar kelimelerle uyumlu biyomedikal mühendisliği başlığı önerildi.'},
    {"title": 'Biomedical Image Registration Using Neural Networks', "tags": ['medical imaging'], "rationale": 'Tıbbi görüntüleme ve biyomedikal doğrulama odağıyla önerildi.'},
    {"title": 'Real-Time ECG Quality Monitoring for Telemedicine Systems', "tags": ['ecg'], "rationale": 'Biyomedikal sinyal işleme ve sinyal kalitesi odağı korunarak önerildi.'},
    {"title": 'Federated Learning in Connected Biomedical Devices', "tags": ['federated learning', 'smart medical devices'], "rationale": 'Akıllı tıbbi cihaz güvenilirliği ve doğrulama odağıyla önerildi.'},
    {"title": 'AI-Assisted Patient Monitoring in Intensive Care Units', "tags": ['patient monitoring'], "rationale": 'Biyomedikal kalibrasyon, hasta monitörü ve sinyal doğrulama odağı korunarak önerildi.'},
    {"title": 'Smart Biomedical Wearables for Elderly Care', "tags": ['wearable health systems'], "rationale": 'Giyilebilir sistemler ve biyosensör bağlamı korunarak önerildi.'},
    {"title": 'Machine Learning for Biomedical Sensor Data Integrity', "tags": ['biomedical engineering'], "rationale": 'Girilen anahtar kelimelerle uyumlu biyomedikal mühendisliği başlığı önerildi.'},
    {"title": 'Deep Learning-Based Respiratory Disease Detection', "tags": ['biomedical engineering'], "rationale": 'Girilen anahtar kelimelerle uyumlu biyomedikal mühendisliği başlığı önerildi.'},
    {"title": 'Explainable AI for Smart Diagnostic Imaging Systems', "tags": ['explainable ai', 'medical imaging'], "rationale": 'Tıbbi görüntüleme ve biyomedikal doğrulama odağıyla önerildi.'},
    {"title": 'Wearable Biosensors for Cardiovascular Monitoring', "tags": ['biosensor', 'wearable health systems'], "rationale": 'Giyilebilir sistemler ve biyosensör bağlamı korunarak önerildi.'},
    {"title": 'Biomedical Signal Reliability Assessment with AI', "tags": ['biomedical signal processing', 'biomedical reliability'], "rationale": 'Biyomedikal sinyal işleme ve sinyal kalitesi odağı korunarak önerildi.'},
    {"title": 'Real-Time EEG-Based Seizure Detection Systems', "tags": ['eeg'], "rationale": 'Biyomedikal sinyal işleme ve sinyal kalitesi odağı korunarak önerildi.'},
    {"title": 'AI-Powered Biomedical Image Denoising Frameworks', "tags": ['medical imaging'], "rationale": 'Tıbbi görüntüleme ve biyomedikal doğrulama odağıyla önerildi.'},
    {"title": 'Smart Medical Device Monitoring Using IoMT', "tags": ['iomt', 'smart medical devices'], "rationale": 'Akıllı tıbbi cihaz güvenilirliği ve doğrulama odağıyla önerildi.'},
    {"title": 'Edge AI for Low-Latency Biomedical Monitoring', "tags": ['edge ai'], "rationale": 'Girilen anahtar kelimelerle uyumlu biyomedikal mühendisliği başlığı önerildi.'},
    {"title": 'Deep Learning for Smart Rehabilitation Robotics', "tags": ['rehabilitation systems'], "rationale": 'Girilen anahtar kelimelerle uyumlu biyomedikal mühendisliği başlığı önerildi.'},
    {"title": 'AI-Based Medical Device Failure Prediction', "tags": ['smart medical devices'], "rationale": 'Akıllı tıbbi cihaz güvenilirliği ve doğrulama odağıyla önerildi.'},
    {"title": 'Explainable Deep Learning in Neuroimaging Analysis', "tags": ['neuroimaging', 'explainable ai', 'medical imaging'], "rationale": 'Alzheimer, nörogörüntüleme veya nörolojik biyomedikal bağlam korunarak önerildi.'},
    {"title": 'Biomedical Signal Fusion for Remote Patient Monitoring', "tags": ['biomedical signal processing', 'patient monitoring', 'remote healthcare'], "rationale": 'Biyomedikal kalibrasyon, hasta monitörü ve sinyal doğrulama odağı korunarak önerildi.'},
    {"title": 'Transformer-Based ECG Pattern Recognition Systems', "tags": ['ecg'], "rationale": 'Biyomedikal sinyal işleme ve sinyal kalitesi odağı korunarak önerildi.'},
    {"title": 'AI-Assisted Smart Sensor Validation for Medical Devices', "tags": ['smart medical devices'], "rationale": 'Akıllı tıbbi cihaz güvenilirliği ve doğrulama odağıyla önerildi.'},
    {"title": 'Wearable Biomedical Systems for Continuous Health Assessment', "tags": ['wearable health systems'], "rationale": 'Giyilebilir sistemler ve biyosensör bağlamı korunarak önerildi.'},
    {"title": 'Deep Learning for Multimodal Medical Imaging Fusion', "tags": ['multimodal fusion', 'medical imaging'], "rationale": 'Tıbbi görüntüleme ve biyomedikal doğrulama odağıyla önerildi.'},
    {"title": 'AI-Based Physiological Monitoring in Smart Healthcare Environments', "tags": ['physiological monitoring'], "rationale": 'Girilen anahtar kelimelerle uyumlu biyomedikal mühendisliği başlığı önerildi.'},
]

CURATED_BIOMEDICAL_TOPIC_BANK = BIOMEDICAL_TOPIC_BANK

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
        status_line = f"PubMed unavailable, OpenAlex active | PubMed: {distribution.get('PubMed', 0)} | OpenAlex: {distribution.get('OpenAlex', 0)}"
    elif has_pubmed and has_openalex:
        status_line = f"Hybrid Intelligence Active | PubMed: {distribution.get('PubMed', 0)} | OpenAlex: {distribution.get('OpenAlex', 0)}"
    elif has_pubmed:
        status_line = f"PubMed Connected | PubMed: {distribution.get('PubMed', 0)}"
    elif has_openalex:
        status_line = f"OpenAlex Connected | OpenAlex: {distribution.get('OpenAlex', 0)}"
    else:
        status_line = "PubMed Ready &nbsp;&nbsp; OpenAlex Ready &nbsp;&nbsp; Hybrid Intelligence Ready"

    logo_src = logo_data_uri()
    hero_logo_html = ""
    if logo_src:
        hero_logo_html = f"""
<div style="text-align:center; margin-bottom: 1rem;">
    <div style="background: rgba(255,255,255,0.94); padding: 12px 18px; border-radius: 16px; display: inline-block; box-shadow: 0 14px 32px rgba(15,23,42,0.18);">
        <img src="{logo_src}" style="width:250px; max-width:82vw; height:auto; display:block;" />
    </div>
</div>"""

    hero_html = f"""
<div class="rm-hero">
    {hero_logo_html}
    <h1>ResearchMind AI</h1>
    <h2>AI-Powered Research Intelligence Platform</h2>
    <div class="rm-badges">
        <span class="rm-badge">Trend Analysis</span>
        <span class="rm-badge">Research Gap Detection</span>
        <span class="rm-badge">AI Topic Recommendation</span>
    </div>
    <div class="rm-card-note" style="margin-top: 0.9rem;">{status_line}</div>
    <p>Araştırma konusunu gir, analiz dönemini seç ve tek tıkla trend, fırsat ve Research Gap Score sonuçlarını üret.</p>
    <p><strong>Bu demo sürüm yalnızca Biyomedikal Mühendisliği alanı için optimize edilmiştir.</strong><br/>
    Analizler; biyosensörler, tıbbi cihazlar, biyomedikal sinyaller, giyilebilir sağlık sistemleri ve tıbbi görüntüleme ekseninde değerlendirilir.</p>
</div>
"""
    st.markdown(hero_html, unsafe_allow_html=True)


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
    replacements.update({
        "Biomedical engineering objects and metrics were preserved from the query.": "Sorgudaki biyomedikal mühendisliği odakları korunarak öneri oluşturuldu.",
        "Uses biomedical engineering validation metrics.": "Biyomedikal mühendisliği doğrulama kriterleri kullanılarak oluşturuldu.",
        "Domain-aware fallback generated from the detected query concepts.": "Sorgudaki biyomedikal odaklar dikkate alınarak öneri oluşturuldu.",
        "Deterministic local fallback generated a complete research topic.": "Biyomedikal mühendisliği odağı korunarak tamamlanmış bir araştırma başlığı üretildi.",
    })
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
        return "Doygun / yüksek rekabet", "low"
    if numeric < 60:
        return "Rekabetçi fakat daraltılabilir", "mid"
    if numeric < 80:
        return "Yükselen araştırma fırsatı", "high"
    return "Güçlü stratejik araştırma fırsatı", "high"

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
    numeric = parse_numeric(score)
    growth_value = parse_numeric(growth)
    if numeric >= 65 and growth_value >= 0:
        return "Yükselen fırsat", "high"
    if numeric >= 40:
        return "Rekabetçi alan", "mid"
    return "Doygun başlık", "low"

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
    text = text.replace("’", "'").replace("`", "'").replace("Â´", "'")
    text = re.sub(r"\balzheimer\?s\b", "alzheimer", text)
    text = re.sub(r"\balzheimer'?s\b", "alzheimer", text)
    text = re.sub(r"\balzheimer\s+s\b", "alzheimer", text)
    text = re.sub(r"\bdisease\b", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def apply_phrase_replacements(text: str, replacements: dict[str, str]) -> str:
    clean = unicodedata.normalize("NFKC", str(text or ""))
    for source, target in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        pattern = rf"(?<![A-Za-z0-9]){re.escape(source)}(?![A-Za-z0-9])"
        clean = re.sub(pattern, target, clean, flags=re.I)
    return re.sub(r"\s+", " ", clean).strip()


def turkishize_report_terms(text: str) -> str:
    replacements = {
        "biomedical signals": "biyomedikal sinyaller",
        "biomedical signal": "biyomedikal sinyal",
        "biosensors": "biyosensörler",
        "biosensor": "biyosensör",
        "wearable health systems": "giyilebilir sağlık sistemleri",
        "wearable health system": "giyilebilir sağlık sistemi",
        "medical devices": "tıbbi cihazlar",
        "medical device": "tıbbi cihaz",
        "medical imaging systems": "tıbbi görüntüleme sistemleri",
        "medical imaging system": "tıbbi görüntüleme sistemi",
        "signal quality": "sinyal kalitesi",
        "device reliability": "cihaz güvenilirliği",
        "diagnostic accuracy": "tanısal doğruluk",
        "measurement precision": "ölçüm hassasiyeti",
        "classification performance": "sınıflandırma performansı",
        "external validation reliability": "dış doğrulama güvenilirliği",
        "model interpretability": "model yorumlanabilirliği",
    }
    clean = str(text or "")
    for source, target in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        clean = re.sub(rf"\b{re.escape(source)}\b", target, clean, flags=re.I)
    clean = re.sub(r"Konuyu ([^;]+), ([^;]+) odağında", r"Konuyu \1 ve \2 odağında", clean)
    return re.sub(r"\s+", " ", clean).strip()


def normalize_keywords_for_domain(text: str) -> dict[str, str]:
    original = re.sub(r"\s+", " ", str(text or "").replace("\n", " ")).strip()
    expanded = apply_phrase_replacements(original, TURKISH_BIOMEDICAL_MAP)
    return {"original": original, "expanded": expanded, "normalized": normalize_topic_key(expanded), "display": original or expanded}

def normalize_field_keywords(text: str) -> str:
    return normalize_keywords_for_domain(text)["normalized"]


def curated_topic_query_text(seed: str) -> str:
    expanded = apply_phrase_replacements(str(seed or ""), {
        **TURKISH_BIOMEDICAL_MAP,
        "biyomedikal kalibrasyon": "biomedical calibration",
        "hastabaşı monitörü": "bedside patient monitor",
        "hasta başı monitörü": "bedside patient monitor",
        "hasta basi monitoru": "bedside patient monitor",
        "hasta monitörü": "patient monitor",
        "hasta monitoru": "patient monitor",
        "sinyal işleme": "signal processing",
        "sinyal isleme": "signal processing",
        "ekg": "ecg",
        "mrg": "mri",
        "mr": "mri",
        "yapay zeka": "artificial intelligence",
        "yapay zekâ": "artificial intelligence",
        "açıklanabilir yapay zeka": "explainable ai",
        "aciklanabilir yapay zeka": "explainable ai",
    })
    return normalize_topic_key(expanded)


def is_bad_suffix_title(title: str, query: str = "") -> bool:
    key = normalize_topic_key(title)
    if not key:
        return True
    bad_suffixes = [
        "for explainable ai",
        "for transformer model",
        "for multimodal learning",
        "for risk prediction",
        "for small dataset learning",
        "for federated learning",
    ]
    if any(key.endswith(suffix) for suffix in bad_suffixes):
        return True
    query_key = normalize_topic_key(query)
    return bool(query_key and key.startswith(query_key + " for "))


def is_near_duplicate_title(title: str, query: str) -> bool:
    title_key = normalize_topic_key(title)
    query_key = normalize_topic_key(query)
    if not title_key or not query_key:
        return False
    if title_key == query_key:
        return True
    if title_key.startswith(query_key) or query_key.startswith(title_key):
        return True

    title_tokens = set(title_key.split())
    query_tokens = set(query_key.split())
    if not title_tokens or not query_tokens:
        return False
    overlap_ratio = len(title_tokens & query_tokens) / max(1, min(len(title_tokens), len(query_tokens)))
    return overlap_ratio > 0.85


TOPIC_RATIONALE_TEXT_FIXES = {
    "Biyomedikal sinyal işleme ve sinyal kalitesi odağı korunarak önerildi.": "Biyomedikal sinyal işleme ve sinyal kalitesi odağı korunarak önerildi.",
    "Giyilebilir sistemler ve biyosensör bağlamı korunarak önerildi.": "Giyilebilir sistemler ve biyosensör bağlamı korunarak önerildi.",
    "Alzheimer, nörogörüntüleme veya nörolojik biyomedikal bağlam korunarak önerildi.": "Alzheimer, nörogörüntüleme veya nörolojik biyomedikal bağlam korunarak önerildi.",
    "Biyomedikal kalibrasyon, hasta monitörü ve sinyal doğrulama odağı korunarak önerildi.": "Biyomedikal kalibrasyon, hasta monitörü ve sinyal doğrulama odağı korunarak önerildi.",
    "Akıllı tıbbi cihaz güvenilirliği ve doğrulama odağıyla önerildi.": "Akıllı tıbbi cihaz güvenilirliği ve doğrulama odağıyla önerildi.",
    "Tıbbi görüntüleme ve biyomedikal doğrulama odağıyla önerildi.": "Tıbbi görüntüleme ve biyomedikal doğrulama odağıyla önerildi.",
    "Girilen anahtar kelimelerle uyumlu biyomedikal mühendisliği başlığı önerildi.": "Girilen anahtar kelimelerle uyumlu biyomedikal mühendisliği başlığı önerildi.",
}


def clean_topic_rationale(text: str) -> str:
    value = str(text or "").strip()
    return TOPIC_RATIONALE_TEXT_FIXES.get(value, value)


def curated_topic_recommendations(seed: str, max_items: int = 5) -> list[dict]:
    query = curated_topic_query_text(seed)
    query_tokens = set(query.split())
    scored = []

    for index, item in enumerate(CURATED_BIOMEDICAL_TOPIC_BANK):
        title = str(item.get("title", ""))
        title_key = normalize_topic_key(title)
        title_tokens = set(title_key.split())
        tags = [normalize_topic_key(tag) for tag in item.get("tags", [])]
        if not any(term in query for term in ["sports", "football", "athlete", "player", "performance"]) and any(
            tag in {"sports", "football", "athlete"} for tag in tags
        ):
            continue
        if "eeg" not in query and any(tag == "eeg" for tag in tags):
            continue
        if "ecg" not in query and any(tag == "ecg" for tag in tags):
            continue
        if not any(term in query for term in ["alzheimer", "dementia", "neurodegenerative", "mri", "neuroimaging"]) and any(
            tag in {"alzheimer", "dementia", "neurodegenerative", "mild cognitive impairment", "neuroimaging"} for tag in tags
        ):
            continue
        score = 0
        for tag in item.get("tags", []):
            tag_key = normalize_topic_key(tag)
            if not tag_key:
                continue
            tag_tokens = set(tag_key.split())
            if re.search(rf"(?<![a-z0-9]){re.escape(tag_key)}(?![a-z0-9])", query):
                score += 3
            elif query_tokens & tag_tokens:
                score += 2
        score += len(query_tokens & title_tokens)
        if score > 0:
            scored.append((score, index, item))

    if scored:
        scored = sorted(scored, key=lambda row: (-row[0], row[1]))[:20]
        rotation = int(st.session_state.get("topic_suggester_rotation", 0)) if hasattr(st, "session_state") else 0
        seed_hash = hashlib.sha256(f"{query}|{datetime.now().date()}|{rotation}".encode("utf-8")).hexdigest()
        scored = sorted(scored, key=lambda row: (-row[0], hashlib.sha256(f"{seed_hash}|{row[1]}".encode("utf-8")).hexdigest()))
        candidates = [item for _, _, item in scored]
    else:
        candidates = []

    general = [
        item for item in CURATED_BIOMEDICAL_TOPIC_BANK
        if any(tag in item.get("tags", []) for tag in ["biomedical signal", "biosensor", "medical device", "wearable"])
    ]
    candidates.extend(general)
    candidates.extend(CURATED_BIOMEDICAL_TOPIC_BANK)

    results = []
    seen = set()
    for item in candidates:
        title = str(item.get("title", "")).strip()
        key = normalize_topic_key(title)
        if not title or key in seen or is_bad_suffix_title(title, seed) or is_near_duplicate_title(title, seed):
            continue
        seen.add(key)
        results.append({
            "title": title,
            "rationale": clean_topic_rationale(
                item.get("rationale") or "Girilen anahtar kelimelerle uyumlu biyomedikal mühendisliği başlığı önerildi."
            ),
        })
        if len(results) >= max_items:
            break
    return results


def simplify_biomedical_retrieval_query(query: str) -> str:
    normalized = normalize_keywords_for_domain(query)["normalized"]
    if not normalized:
        return ""

    if len(normalized.split()) > 10 and "biosensor" in normalized and "wearable" in normalized:
        return "wearable biosensor physiological signal monitoring"

    priority_terms = [
        "alzheimer",
        "dementia",
        "cognitive decline",
        "mild cognitive impairment",
        "neurodegenerative",
        "eeg",
        "ecg",
        "emg",
        "mri",
        "mr",
        "ct",
        "pet",
        "neuroimaging",
        "medical imaging",
        "biosensor",
        "wearable biosensor",
        "wearable",
        "biomedical signal",
        "physiological signal",
        "signal quality",
        "artifact reduction",
        "noise reduction",
        "sensor fusion",
        "medical device",
        "health monitoring",
        "diagnosis",
        "classification",
        "machine learning",
        "deep learning",
        "explainable ai",
        "artificial intelligence",
    ]
    selected = []
    for term in priority_terms:
        key = normalize_topic_key(term)
        if key and re.search(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", normalized):
            selected.append(term)

    if len(normalized.split()) > 10 and selected:
        return safe_join(dict.fromkeys(selected[:7]).keys(), " ")

    return normalized


def migrate_legacy_field(value: str) -> str:
    return BIOMEDICAL_FIELD

def get_domain_family(selected_field: str) -> str:
    return "biomedical_engineering"

def is_engineering_field(selected_field: str) -> bool:
    return True

def is_healthcare_field(selected_field: str) -> bool:
    return False

def compatibility_domain(selected_field: str) -> str:
    return ENGINEERING_DOMAIN

def current_selected_field() -> str:
    value = st.session_state.get("selected_research_field", BIOMEDICAL_FIELD)
    return BIOMEDICAL_FIELD if value != BIOMEDICAL_FIELD else value

def compute_keyword_score(normalized_topic: str, keywords: list[str]) -> tuple[float, list[str]]:
    score = 0.0
    matches: list[str] = []
    tokens = set(normalized_topic.split())
    for keyword in keywords:
        key = normalize_topic_key(keyword)
        if not key:
            continue
        if re.search(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", normalized_topic):
            score += 3
            matches.append(keyword)
        elif tokens & set(key.split()):
            score += 1
            matches.append(keyword)
    return score, list(dict.fromkeys(matches))


def supported_field_intent(user_topic: str, selected_field: str | None = None) -> dict:
    selected_field = BIOMEDICAL_FIELD
    ontology = BIOMEDICAL_ENGINEERING_ONTOLOGY
    normalized = normalize_keywords_for_domain(user_topic)
    score, matches = compute_keyword_score(normalized["normalized"], ontology["keywords"])
    preferred_objects = list(ontology.get("preferred_objects", []))
    preferred_metrics = list(ontology.get("preferred_metrics", []))
    key = normalized["normalized"]
    priority_objects = []
    priority_metrics = []
    neurodegenerative_terms = [
        "alzheimer",
        "dementia",
        "cognitive decline",
        "mild cognitive impairment",
        "neurodegenerative",
    ]
    has_neurodegenerative_context = any(term in key for term in neurodegenerative_terms)
    if has_neurodegenerative_context:
        if "eeg" in key:
            priority_objects.append("EEG signals")
        if "mri" in key or "mr" in key or "neuroimaging" in key:
            priority_objects.append("MRI neuroimaging data")
        priority_objects.extend([
            "Alzheimer diagnosis workflows",
            "multimodal clinical and imaging datasets",
            "cognitive assessment data",
            "neurodegenerative disease screening systems",
        ])
        if "explainable" in key or "xai" in key:
            priority_metrics.append("model interpretability")
        priority_metrics.extend([
            "diagnostic accuracy",
            "early detection performance",
            "clinical validation quality",
            "classification performance",
            "external validation reliability",
        ])
    if "biosensor" in key:
        priority_objects.extend(["biosensors", "wearable health systems"])
    if "wearable" in key:
        priority_objects.append("wearable health systems")
    if "medical device" in key:
        priority_objects.append("medical devices")
    if "biomedical signal" in key or "ecg" in key or "eeg" in key or "emg" in key:
        priority_objects.append("biomedical signals")
    if "medical imaging" in key or "mri" in key or "ct" in key:
        priority_objects.append("medical imaging systems")
    if "signal quality" in key:
        priority_metrics.insert(0, "signal quality")
    if "noise" in key:
        priority_metrics.append("noise robustness")
    if "artifact" in key:
        priority_metrics.append("artifact reduction")
    if "sensor fusion" in key:
        priority_metrics.append("device reliability")
    preferred_objects = list(dict.fromkeys([*priority_objects, *preferred_objects]))
    preferred_metrics = list(dict.fromkeys([*priority_metrics, *preferred_metrics]))
    return {
        "selected_field": BIOMEDICAL_FIELD,
        "selected_domain": ENGINEERING_DOMAIN,
        "domain_family": "biomedical_engineering",
        "subdomain_key": "biomedical_engineering",
        "subdomain_label": BIOMEDICAL_FIELD,
        "confidence": round(min(1.0, score / 9.0), 3),
        "keyword_score": score,
        "matched_keywords": matches,
        "preferred_objects": preferred_objects,
        "preferred_metrics": preferred_metrics,
        "validation_language": ontology.get("validation_language", ""),
        "original_keywords": normalized["original"],
        "expanded_keywords": normalized["expanded"],
        "normalized_keywords": normalized["normalized"],
    }

def _score_subdomain(normalized_text: str, ontology_item: dict) -> tuple[float, list[str]]:
    tokens = set(normalized_text.split())
    score = 0.0
    matches = []
    for keyword in ontology_item.get("keywords", []):
        key = normalize_topic_key(keyword)
        if not key:
            continue
        if re.search(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", normalized_text):
            score += 3
            matches.append(keyword)
            continue
        if tokens & set(key.split()):
            score += 1
            matches.append(keyword)
    return score, list(dict.fromkeys(matches))


def detect_subdomain(normalized_keywords: str, ontology: dict, fallback_key: str) -> dict:
    best_key = fallback_key
    best_score = -1.0
    best_matches: list[str] = []
    for subdomain_key, item in ontology.items():
        if subdomain_key == fallback_key:
            continue
        score, matches = _score_subdomain(normalized_keywords, item)
        if score > best_score:
            best_key = subdomain_key
            best_score = score
            best_matches = matches
    if best_score <= 0:
        best_key = fallback_key
        best_score = 0.0
        best_matches = []
    item = ontology[best_key]
    return {
        "subdomain_key": best_key,
        "subdomain_label": item["subdomain_label"],
        "confidence": round(min(1.0, best_score / 9.0), 3),
        "matched_keywords": best_matches,
        "preferred_objects": list(item.get("preferred_objects", [])),
        "preferred_metrics": list(item.get("preferred_metrics", [])),
    }


def detect_engineering_subdomain(normalized_keywords: str) -> dict:
    return detect_subdomain(normalized_keywords, ENGINEERING_SUBDOMAIN_ONTOLOGY, "general_engineering")


def detect_healthcare_subdomain(normalized_keywords: str) -> dict:
    return detect_subdomain(normalized_keywords, HEALTHCARE_SUBDOMAIN_ONTOLOGY, "general_healthcare")


def domain_intent(query: str, selected_domain: str | None = None) -> dict:
    selected_domain = selected_domain or current_selected_field()
    if selected_domain in SUPPORTED_FIELDS or selected_domain in LEGACY_DOMAIN_TO_FIELD:
        return supported_field_intent(query, migrate_legacy_field(selected_domain))
    processed = normalize_keywords_for_domain(query)
    detected = (
        detect_engineering_subdomain(processed["normalized"])
        if selected_domain == ENGINEERING_DOMAIN
        else detect_healthcare_subdomain(processed["normalized"])
    )
    return {
        **detected,
        "selected_domain": selected_domain,
        "original_keywords": processed["original"],
        "expanded_keywords": processed["expanded"],
        "normalized_keywords": processed["normalized"],
    }


def _cycle_pick(items: list[str], index: int, fallback: str) -> str:
    return items[index % len(items)] if items else fallback


def generate_intent_titles(query: str, selected_domain: str | None = None, min_count: int = 5) -> list[dict[str, str]]:
    return curated_topic_recommendations(query, max_items=min_count)

def clean_topic_label(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "").replace("’", "'")).strip(" ,;")
    text = re.sub(r"\bAlzheimer\?s disease\b", "Alzheimer's disease", text, flags=re.I)
    text = re.sub(r"\bAlzheimer's disease\b", "Alzheimer's disease", text, flags=re.I)
    return text


def current_selected_domain() -> str:
    return compatibility_domain(current_selected_field())


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


def classify_biomedical_topic_input(user_topic: str) -> dict:
    expanded = apply_phrase_replacements(str(user_topic or ""), {**TURKISH_BIOMEDICAL_MAP, **TURKISH_UNRELATED_MAP})
    normalized_topic = normalize_topic_key(expanded)
    biomedical_terms = matched_domain_terms(normalized_topic, BIOMEDICAL_TERMS)
    strong_biomedical_objects = matched_domain_terms(normalized_topic, STRONG_BIOMEDICAL_OBJECTS)
    general_terms = matched_domain_terms(normalized_topic, GENERAL_RELATED_TERMS)
    unrelated_terms = matched_domain_terms(normalized_topic, UNRELATED_BLOCK_TERMS)

    if unrelated_terms and not biomedical_terms:
        status = "hard_block"
        message = BIOMEDICAL_HARD_BLOCK_MESSAGE
    elif biomedical_terms or strong_biomedical_objects:
        status = "allow"
        message = BIOMEDICAL_MIXED_TERM_WARNING if unrelated_terms else ""
    else:
        status = "soft_guidance"
        message = BIOMEDICAL_SOFT_GUIDANCE_MESSAGE

    return {
        "status": status,
        "matched_biomedical_terms": sorted(set([*biomedical_terms, *strong_biomedical_objects])),
        "strong_biomedical_objects": strong_biomedical_objects,
        "general_terms": general_terms,
        "unrelated_terms": unrelated_terms,
        "message": message,
        "normalized_topic": normalized_topic,
    }


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
    selected_field = BIOMEDICAL_FIELD
    classification = classify_biomedical_topic_input(query)
    normalized_topic = classification["normalized_topic"]
    intent = supported_field_intent(query, selected_field)
    debug = {
        "selected_field": selected_field,
        "selected_domain": ENGINEERING_DOMAIN,
        "domain_family": "biomedical_engineering",
        "normalized_topic": normalized_topic,
        "validation_result": classification["status"],
        "field_score": intent["keyword_score"],
        "matched_keywords": classification["matched_biomedical_terms"],
        "general_terms": classification["general_terms"],
        "unrelated_terms": classification["unrelated_terms"],
        "classification_confidence": "high" if intent["keyword_score"] >= 3 else "low",
    }
    if classification["status"] == "hard_block":
        debug["leakage_terms"] = classification["unrelated_terms"]
        return False, classification["message"], debug
    return True, classification["message"], debug

def validate_supported_field_topic(selected_field: str, user_topic: str) -> tuple[bool, str, dict]:
    return validate_domain_query(user_topic, selected_field)


def forbidden_terms_for_domain(selected_domain: str, query: str) -> set[str]:
    if selected_domain == HEALTHCARE_DOMAIN:
        return HEALTHCARE_BLACKLIST
    if selected_domain == ENGINEERING_DOMAIN and not is_engineering_health_hybrid(query):
        return ENGINEERING_BIOMED_BLACKLIST
    return set()


def domain_specific_strategy(query: str, selected_domain: str) -> dict[str, str]:
    intent = domain_intent(query, selected_domain)
    compat = intent["selected_domain"]
    label = intent["subdomain_label"]
    objects = intent.get("preferred_objects", [])
    metrics = intent.get("preferred_metrics", [])
    object_text = safe_join(objects[:3]) or title_case_topic(intent.get("expanded_keywords") or query)
    metric_text = safe_join(metrics[:3]) or "performance"
    if compat == ENGINEERING_DOMAIN:
        return {
            "direction": f"{label} research focused on {metric_text} in {object_text}",
            "methodology": engineering_methodology_for_intent(intent),
            "evidence": evidence_focus_for_intent(intent),
            "differentiation": f"Differentiate the work through subdomain-specific {metric_text}, benchmark validation, transparent error analysis, and a clearly defined engineering object: {object_text}.",
        }
    if compat == HEALTHCARE_DOMAIN and intent["subdomain_key"] not in {"general_healthcare", "medicine_clinical"}:
        return {
            "direction": f"{label} research focused on {metric_text} in {object_text}",
            "methodology": healthcare_methodology_for_intent(intent),
            "evidence": evidence_focus_for_intent(intent),
            "differentiation": f"Differentiate the work through {label.lower()}-specific outcomes, validated measurement instruments, transparent error analysis, and a clearly defined evidence base.",
        }
    key = normalize_topic_key(query)
    if compat == ENGINEERING_DOMAIN:
        if any(term in key for term in ["seismic", "earthquake", "structural", "bridge", "building", "infrastructure"]):
            return {
                "direction": "Explainable AI for seismic vulnerability and infrastructure resilience assessment",
                "methodology": "structural health monitoring; finite element analysis; sensor fusion; seismic vulnerability modeling; reliability analysis",
                "evidence": "structural sensor streams; ground-motion records; finite element simulations; benchmark infrastructure datasets",
                "differentiation": "Differentiate the work through engineering performance metrics, benchmarked seismic scenarios, and interpretable resilience indicators.",
            }
        if any(term in key for term in ["cfd", "thermal", "heat transfer", "thermodynamics"]):
            return {
                "direction": "Simulation-driven thermal optimization using CFD and interpretable surrogate modeling",
                "methodology": "CFD simulation; finite element analysis; optimization algorithms; surrogate modeling; benchmark validation",
                "evidence": "simulation outputs; thermal boundary conditions; experimental benchmark datasets; engineering performance metrics",
                "differentiation": "Differentiate the work through reproducible simulation protocols, optimization constraints, and validated thermal performance gains.",
            }
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
                "methodology": "digital twin modeling; fault detection; time-series forecasting; sensor fusion; reliability analysis",
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
    intent = domain_intent(query, selected_domain)
    compat = intent["selected_domain"]
    label = intent["subdomain_label"]
    objects = safe_join(intent.get("preferred_objects", [])[:3]) or "the target research object"
    metrics = safe_join(intent.get("preferred_metrics", [])[:3]) or "domain-specific performance"
    if compat == ENGINEERING_DOMAIN:
        return (
            f"This {label} topic should be framed around {objects}, measurable {metrics}, "
            "benchmark or simulation evidence, and engineering performance metrics. Strong differentiation "
            "comes from preserving the engineering object, deployment feasibility, and robust validation."
        )
    return (
        f"This {label} topic should be framed around {objects}, measurable {metrics}, "
        "subdomain-specific evidence, and transparent validation. Strong differentiation comes from preserving "
        "the healthcare research object rather than drifting into unrelated technical suffixes."
    )


def domain_specific_paperability_reason(query: str, selected_domain: str) -> str:
    intent = domain_intent(query, selected_domain)
    compat = intent["selected_domain"]
    label = intent["subdomain_label"]
    objects = safe_join(intent.get("preferred_objects", [])[:2]) or "domain-specific evidence"
    metrics = safe_join(intent.get("preferred_metrics", [])[:2]) or "validated outcomes"
    if compat == ENGINEERING_DOMAIN:
        return f"{label} evidence such as {objects}, benchmark datasets, and {metrics} improves publication feasibility."
    return f"{label} evidence such as {objects} and {metrics} improves publication feasibility."


def sanitize_engineering_language(text: str) -> str:
    clean = str(text or "")
    for source, target in sorted(ENGINEERING_LANGUAGE_REPLACEMENTS.items(), key=lambda item: len(item[0]), reverse=True):
        pattern = rf"(?<![A-Za-z0-9]){re.escape(source)}(?![A-Za-z0-9])"
        clean = re.sub(pattern, target, clean, flags=re.I)
    return clean


def sanitize_biomedical_text(text: str) -> str:
    clean = str(text or "")
    replacements = {
        "clinical cohorts": "benchmark biomedical datasets",
        "patient groups": "biomedical signal datasets",
        "treatment response": "device performance response",
        "prognostic performance": "classification performance",
        "prognosis": "performance prediction",
        "mortality risk": "device risk indicator",
        "readmission risk": "system reliability risk",
        "clinical outcome prediction": "biomedical system performance prediction",
        "clinically meaningful endpoints": "biomedical engineering performance metrics",
        "external clinical evidence": "external biomedical engineering validation evidence",
    }
    clean = apply_phrase_replacements(clean, replacements)
    for term in BIOMEDICAL_CLINICAL_DRIFT_TERMS:
        clean = re.sub(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", "", clean, flags=re.I)
    return re.sub(r"\s+", " ", clean).strip(" ;,.")


def forbidden_terms_for_intent(intent: dict) -> set[str]:
    explicit_text = intent.get("normalized_keywords", "")
    forbidden = {
        "bridge", "concrete", "seismic", "earthquake", "coating", "corrosion", "uav", "drone",
        "tax", "fiscal", "economics", "public expenditure", "nursing", "midwifery",
        "clinical cohorts", "patient groups", "treatment response", "prognostic performance",
        "mortality risk", "readmission risk",
    }
    return {term for term in forbidden if normalize_topic_key(term) not in explicit_text}

def find_forbidden_terms(text: str, forbidden: set[str]) -> list[str]:
    key = normalize_topic_key(text)
    return sorted(term for term in forbidden if normalize_topic_key(term) and re.search(rf"(?<![a-z0-9]){re.escape(normalize_topic_key(term))}(?![a-z0-9])", key))


def sanitize_text_for_intent(text: str, intent: dict) -> str:
    clean = sanitize_biomedical_text(text)
    forbidden = forbidden_terms_for_intent(intent)
    for term in find_forbidden_terms(clean, forbidden):
        clean = re.sub(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", "", clean, flags=re.I)
    return re.sub(r"\s+", " ", clean).strip(" ;,.")

def sanitize_suggestions_for_intent(suggestions: pd.DataFrame, query: str, selected_domain: str | None = None) -> pd.DataFrame:
    suggestions = _as_dataframe(suggestions)
    intent = domain_intent(query, selected_domain)
    forbidden = forbidden_terms_for_intent(intent)
    title_col = "suggested_research_topic" if "suggested_research_topic" in suggestions.columns else "suggested_topic" if "suggested_topic" in suggestions.columns else None

    if suggestions.empty or not title_col:
        return intent_topics_to_dataframe(query, intent["selected_domain"])

    out = suggestions.copy()
    before = len(out)
    mask = out.astype(str).agg(" ".join, axis=1).map(lambda text: not find_forbidden_terms(text, forbidden))
    out = out.loc[mask].copy()
    if len(out) < 3:
        out = pd.concat([out, intent_topics_to_dataframe(query, intent["selected_domain"])], ignore_index=True, sort=False)
    out[title_col] = out[title_col].map(lambda value: sanitize_text_for_intent(value, intent))
    out = out[out[title_col].fillna("").astype(str).str.strip().ne("")]
    out = out.drop_duplicates(subset=[title_col], keep="first").head(8).reset_index(drop=True)
    out.attrs["domain_filtered_count"] = max(0, before - len(out))
    return out


def sanitize_topic_items_for_intent(topics: list[dict[str, str]], query: str, selected_domain: str | None = None) -> list[dict[str, str]]:
    intent = domain_intent(query, selected_domain)
    forbidden = forbidden_terms_for_intent(intent)
    clean_topics = []
    seen = set()
    for item in topics:
        title = sanitize_text_for_intent(item.get("title", ""), intent)
        rationale = sanitize_text_for_intent(item.get("rationale", ""), intent)
        key = normalize_topic_key(title)
        if title and key not in seen and not find_forbidden_terms(title, forbidden):
            seen.add(key)
            clean_topics.append({"title": title, "rationale": rationale or f"{intent['subdomain_label']} intent preserved."})
    if len(clean_topics) < 3:
        for item in generate_intent_titles(query, intent["selected_domain"], min_count=5):
            key = normalize_topic_key(item["title"])
            if key not in seen:
                seen.add(key)
                clean_topics.append(item)
            if len(clean_topics) >= 5:
                break
    return clean_topics[:5]


def engineering_methodology_for_intent(intent: dict) -> str:
    subdomain = intent.get("subdomain_key")
    pools = {
        "civil_structural_engineering": ["structural health monitoring", "finite element analysis", "fragility analysis", "sensor fusion", "reliability analysis"],
        "materials_engineering": ["microstructure-property modeling", "durability testing", "materials informatics", "mechanical performance prediction", "experimental benchmark validation"],
        "mechanical_engineering": ["CFD/FEM simulation", "thermal optimization", "vibration analysis", "surrogate modeling", "benchmark validation"],
        "electrical_electronics_engineering": ["signal processing", "embedded systems validation", "control systems", "IoT/edge AI architecture", "reliability analysis"],
        "computer_software_engineering": ["benchmark dataset evaluation", "robustness testing", "scalability analysis", "privacy-aware architecture", "computational efficiency analysis"],
        "aerospace_uav_engineering": ["flight control modeling", "sensor fusion", "edge AI", "trajectory optimization", "real-time monitoring"],
        "chemical_process_engineering": ["process optimization", "reaction kinetics modeling", "mass transfer analysis", "simulation-based validation", "stability analysis"],
        "industrial_systems_engineering": ["operations research", "simulation modeling", "scheduling optimization", "decision support modeling", "quality control analytics"],
        "energy_environmental_engineering": ["digital twin modeling", "predictive maintenance", "sensor fusion", "time-series forecasting", "reliability analysis"],
        "biomedical_engineering": ["biomedical signal processing", "device reliability testing", "usability validation", "safety analysis", "benchmark biomedical datasets"],
    }
    return "; ".join(pools.get(subdomain, ["simulation-based validation", "optimization algorithms", "benchmark dataset evaluation", "real-time monitoring", "reliability analysis"]))


def healthcare_methodology_for_intent(intent: dict) -> str:
    subdomain = intent.get("subdomain_key")
    pools = {
        "nursing": ["validated nursing scales", "care quality modeling", "workload analysis", "patient safety outcome modeling", "transparent error analysis"],
        "midwifery": ["maternal risk stratification", "prenatal care indicators", "birth outcome modeling", "neonatal safety assessment", "transparent error analysis"],
        "biomedical_engineering_health": ["biomedical signal processing", "device reliability testing", "usability validation", "safety assessment", "benchmark biomedical datasets"],
        "public_health": ["epidemiological modeling", "population-level validation", "intervention effectiveness analysis", "bias assessment", "uncertainty analysis"],
        "dentistry": ["dental imaging analytics", "oral health outcome modeling", "periodontal risk assessment", "implant stability analysis", "transparent error analysis"],
        "pharmacy_pharmacology": ["pharmacovigilance modeling", "drug safety analysis", "dosage optimization", "adverse reaction risk prediction", "transparent error analysis"],
        "nutrition_dietetics": ["dietary record analysis", "nutritional status modeling", "metabolic outcome assessment", "adherence modeling", "transparent error analysis"],
        "physiotherapy_rehabilitation": ["gait analysis", "functional recovery modeling", "exercise intervention assessment", "rehabilitation outcome validation", "transparent error analysis"],
        "mental_health_psychology": ["validated psychological scales", "symptom severity modeling", "behavioral data analysis", "treatment response prediction", "transparent error analysis"],
    }
    return "; ".join(pools.get(subdomain, ["external validation", "clinically interpretable modeling", "transparent error analysis", "outcome-focused evaluation"]))


def evidence_focus_for_intent(intent: dict) -> str:
    subdomain = intent.get("subdomain_key")
    engineering = {
        "civil_structural_engineering": "benchmark structural datasets; simulation-based validation; fragility analysis; structural safety metrics",
        "materials_engineering": "experimental datasets; microstructure-property validation; mechanical performance metrics; durability testing",
        "mechanical_engineering": "CFD/FEM outputs; thermal or vibration benchmarks; experimental measurements; engineering performance metrics",
        "electrical_electronics_engineering": "sensor/network measurements; embedded system logs; signal quality benchmarks; latency and reliability metrics",
        "computer_software_engineering": "benchmark datasets; system logs; scalability tests; security and robustness metrics",
        "aerospace_uav_engineering": "UAV telemetry; flight-control logs; swarm communication benchmarks; real-time performance metrics",
        "chemical_process_engineering": "reactor/process data; catalyst measurements; mass transfer benchmarks; process stability metrics",
        "industrial_systems_engineering": "production logs; logistics datasets; simulation models; productivity and throughput metrics",
        "energy_environmental_engineering": "SCADA signals; turbine or battery sensor streams; environmental monitoring data; reliability metrics",
        "biomedical_engineering": "device measurements; biomedical signals; usability tests; safety and reliability evidence",
    }
    healthcare = {
        "medicine_clinical": "external validation; clinically meaningful endpoints; transparent error analysis; clinically interpretable model outputs",
        "nursing": "validated nursing scales; care quality indicators; patient safety outcomes; workload measures",
        "midwifery": "maternal and neonatal outcomes; prenatal care indicators; birth outcome measures; transparent risk stratification",
        "biomedical_engineering_health": "device reliability; signal quality; benchmark biomedical datasets; usability and safety evidence",
        "public_health": "population-level datasets; epidemiological validity; intervention outcomes; bias assessment",
        "dentistry": "dental images; oral health datasets; periodontal assessments; implant outcome evidence",
        "pharmacy_pharmacology": "pharmacovigilance databases; prescription records; drug safety outcomes; dosage response evidence",
        "nutrition_dietetics": "dietary records; nutrition programs; metabolic health profiles; adherence measures",
        "physiotherapy_rehabilitation": "gait signals; exercise intervention records; functional recovery measures; rehabilitation outcomes",
        "mental_health_psychology": "validated psychological scales; behavioral datasets; symptom severity measures; therapy response evidence",
    }
    return engineering.get(subdomain) or healthcare.get(subdomain) or "; ".join((intent.get("preferred_objects", []) + intent.get("preferred_metrics", []))[:5])


def validation_strategy_for_intent(intent: dict) -> str:
    return evidence_focus_for_intent(intent) + "; transparent error analysis."


def _alzheimer_narrowing_text(query: str) -> str:
    key = normalize_topic_key(query)
    evidence = "EEG/MRI veya multimodal nörogörüntüleme verileri"
    if "eeg" in key and not any(term in key for term in ["mri", "mr", "neuroimaging"]):
        evidence = "EEG sinyalleri"
    elif any(term in key for term in ["mri", "mr", "neuroimaging"]) and "eeg" not in key:
        evidence = "MRI/nörogörüntüleme verileri"
    xai = "açıklanabilir yapay zekâ ve yorumlanabilir model çıktıları" if ("explainable" in key or "xai" in key) else "açıklanabilir yapay zekâ"
    return (
        f"Konuyu Alzheimer tanısı, {evidence}, {xai}, "
        "erken tanı performansı ve dış doğrulama protokolü etrafında daraltın."
    )


def is_alzheimer_context(query: str) -> bool:
    key = normalize_topic_key(query)
    return any(term in key for term in [
        "alzheimer", "dementia", "cognitive decline",
        "mild cognitive impairment", "neurodegenerative"
    ])


def domain_narrowing_for_selected(query: str, selected_domain: str) -> str:
    if is_alzheimer_context(query):
        return _alzheimer_narrowing_text(query)

    intent = domain_intent(query, selected_domain)
    object_text = safe_join(intent.get("preferred_objects", [])[:2]) or title_case_topic(intent.get("expanded_keywords") or query)
    metric_text = safe_join(intent.get("preferred_metrics", [])[:2]) or "ölçülebilir performans"
    return turkishize_report_terms(
        f"Konuyu {object_text} odağında; {metric_text}, açık doğrulama protokolü "
        "ve karşılaştırılabilir performans ölçütleriyle daraltın."
    )

def apply_domain_guard_to_results(results: dict) -> dict:
    selected_field = migrate_legacy_field(results.get("selected_field", current_selected_field()))
    selected_domain = compatibility_domain(selected_field)
    query = results.get("query", "")
    intent = domain_intent(query, selected_field)
    forbidden = forbidden_terms_for_intent(intent) | forbidden_terms_for_domain(selected_domain, query)
    corrected = 0
    leakage_terms: set[str] = set()

    def has_forbidden(text: str) -> bool:
        key = normalize_topic_key(text)
        found = {term for term in forbidden if normalize_topic_key(term) and re.search(rf"(?<![a-z0-9]){re.escape(normalize_topic_key(term))}(?![a-z0-9])", key)}
        leakage_terms.update(found)
        return bool(found)

    suggestions = _as_dataframe(results.get("ai_topic_suggestions"))
    if selected_domain in {ENGINEERING_DOMAIN, HEALTHCARE_DOMAIN}:
        results["research_strategy"] = domain_specific_strategy(query, selected_field)
        corrected += 1

    if not suggestions.empty and forbidden:
        title_col = "suggested_research_topic" if "suggested_research_topic" in suggestions.columns else "suggested_topic" if "suggested_topic" in suggestions.columns else None
        if title_col:
            mask = suggestions.astype(str).agg(" ".join, axis=1).map(lambda text: not has_forbidden(text))
            filtered = suggestions.loc[mask].copy()
            corrected += len(suggestions) - len(filtered)
            if len(filtered) < 3:
                fallback = intent_topics_to_dataframe(query, selected_field)
                filtered = pd.concat([filtered, fallback], ignore_index=True).drop_duplicates(subset=[title_col], keep="first")
            results["ai_topic_suggestions"] = sanitize_suggestions_for_intent(filtered.head(8), query, selected_field)

    results["ai_research_insight"] = sanitize_text_for_intent(domain_specific_insight(query, selected_field), intent)
    if forbidden and has_forbidden(results.get("ai_research_insight", "")):
        results["ai_research_insight"] = domain_specific_insight(query, selected_field)
        corrected += 1

    strategy = dict(results.get("research_strategy") or {})
    if forbidden and any(has_forbidden(value) for value in strategy.values()):
        results["research_strategy"] = domain_specific_strategy(query, selected_field)
        corrected += 1

    paperability = dict(results.get("paperability_score") or {})
    if paperability:
        reasons = [reason for reason in paperability.get("reasons", []) if not has_forbidden(reason)]
        if len(reasons) != len(paperability.get("reasons", [])):
            corrected += len(paperability.get("reasons", [])) - len(reasons)
        domain_reason = domain_specific_paperability_reason(query, selected_field)
        if domain_reason not in reasons:
            reasons.insert(0, domain_reason)
        if not reasons:
            reasons = [domain_reason]
        paperability["reasons"] = reasons[:5]
        if selected_domain in {ENGINEERING_DOMAIN, HEALTHCARE_DOMAIN} or has_forbidden(paperability.get("recommended_next_action", "")):
            paperability["recommended_next_action"] = domain_narrowing_for_selected(query, selected_field)
            corrected += 1
        paperability["reasons"] = [
            turkishize_report_terms(sanitize_text_for_intent(reason, intent))
            for reason in paperability["reasons"]
        ]
        paperability["recommended_next_action"] = turkishize_report_terms(
            sanitize_text_for_intent(paperability.get("recommended_next_action", ""), intent)
        )
        metrics = []
        for metric in paperability.get("metrics", []):
            item = dict(metric)
            item["metric"] = sanitize_text_for_intent(item.get("metric", ""), intent)
            if selected_domain == ENGINEERING_DOMAIN and item["metric"].lower().startswith("engineering / practical"):
                item["metric"] = "Engineering / Practical Relevance"
            item["comment"] = sanitize_text_for_intent(item.get("comment", ""), intent)
            metrics.append(item)
        paperability["metrics"] = metrics
        results["paperability_score"] = paperability

    strategy = {key: sanitize_text_for_intent(value, intent) for key, value in dict(results.get("research_strategy") or {}).items()}
    results["research_strategy"] = strategy

    inferred = infer_research_domain(query)
    leakage_score = round(min(1.0, len(leakage_terms) / 5), 2)
    guard = {
        "selected_domain": selected_domain,
        "selected_field": selected_field,
        "inferred_domain": inferred,
        "domain_match": inferred in {selected_domain, "Healthcare-Engineering Hybrid", "Not detected"},
        "leakage_terms": sorted(leakage_terms),
        "corrected_items_count": corrected,
        "domain_leakage_risk_score": leakage_score,
    }
    results["domain_guard"] = guard
    domain_reasoning = dict(results.get("domain_reasoning") or {})
    domain_reasoning["selected_domain"] = selected_domain
    domain_reasoning["selected_field"] = selected_field
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
    selected_field = current_selected_field()
    selected_domain = compatibility_domain(selected_field)
    intent = domain_intent(query, selected_field)
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
        "clinical_domain": intent["subdomain_label"],
        "primary_domain": intent["subdomain_label"],
        "selected_field": selected_field,
        "selected_domain": selected_domain,
        "subdomain_key": intent["subdomain_key"],
        "subdomain_label": intent["subdomain_label"],
        "subdomain_confidence": intent["confidence"],
        "matched_keywords": safe_join(intent.get("matched_keywords", [])[:8]),
        "preferred_objects": safe_join(intent.get("preferred_objects", [])[:6]),
        "preferred_metrics": safe_join(intent.get("preferred_metrics", [])[:6]),
        "domain_consistency_score": round(consistency, 3),
        "domain_consistency": consistency_label,
        "semantic_leakage_risk": risk,
        "leakage_filtered_count": leakage_filtered,
        "concepts": concepts,
    }


def domain_reasoning_to_dataframe(reasoning: dict | None) -> pd.DataFrame:
    reasoning = reasoning or {}
    rows = [
        ("selected subdomain", reasoning.get("subdomain_label", reasoning.get("primary_domain", "-"))),
        ("subdomain confidence", reasoning.get("subdomain_confidence", "-")),
        ("matched keywords", reasoning.get("matched_keywords", "-")),
        ("preferred objects", reasoning.get("preferred_objects", "-")),
        ("preferred metrics", reasoning.get("preferred_metrics", "-")),
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


def build_research_strategy(query: str, suggestions: pd.DataFrame | None = None, selected_domain: str | None = None) -> dict[str, str]:
    selected_domain = selected_domain or current_selected_field()
    intent = domain_intent(query, selected_domain)
    compat = intent["selected_domain"]
    if compat == ENGINEERING_DOMAIN or (compat == HEALTHCARE_DOMAIN and intent["subdomain_key"] not in {"general_healthcare", "medicine_clinical"}):
        return domain_specific_strategy(query, selected_domain)
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
    if is_alzheimer_context(query):
        return _alzheimer_narrowing_text(query)

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
    selected_field = domain_reasoning.get("selected_field", current_selected_field())
    selected_domain = compatibility_domain(selected_field)
    intent = domain_intent(query, selected_field)
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

    if selected_domain == ENGINEERING_DOMAIN:
        method_terms = ["explainable", "xai", "digital twin", "sensor fusion", "predictive maintenance", "simulation", "finite element", "cfd", "optimization", "anomaly detection", "edge ai", "embedded", "control systems", "reliability analysis"]
        clinical_terms = intent.get("preferred_objects", []) + intent.get("matched_keywords", [])
        evidence_terms = intent.get("preferred_objects", []) + intent.get("preferred_metrics", []) + ["benchmark", "simulation", "sensor", "dataset", "validation"]
        differentiation_terms = intent.get("preferred_metrics", []) + ["benchmark", "real time", "reliability", "optimization", "transparent error analysis"]
    else:
        method_terms = ["explainable", "xai", "federated", "multimodal", "transformer", "vision transformer", "clinical validation", "mri", "pet"]
        clinical_terms = ["alzheimer", "cancer", "diagnosis", "clinical decision support", "medical imaging", "neuroimaging", "patient", "disease", *intent.get("matched_keywords", [])]
        evidence_terms = ["mri", "pet", "eeg", "public dataset", "clinical records", "small dataset", "multi center", "multi-center", "validation", *intent.get("preferred_objects", [])]
    differentiation_terms = ["explainability", "açıklanabilirlik", "privacy", "gizlilik", "clinical decision support", "multimodal fusion", "federated learning", "federated"]

    if selected_domain == ENGINEERING_DOMAIN:
        differentiation_terms = intent.get("preferred_metrics", []) + ["benchmark", "real time", "reliability", "optimization", "transparent error analysis"]
    else:
        differentiation_terms = ["explainability", "privacy", "clinical decision support", "multimodal fusion", "federated learning", "federated", *intent.get("preferred_metrics", [])]

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
        reasons.append("Güçlü yöntem unsurları mevcut: " + safe_join(dict.fromkeys(method_hits[:4]).keys()) + ".")
    if clinical_hits:
        reasons.append("Klinik/pratik bağlam SCI yayın potansiyelini destekliyor.")
    if evidence_hits:
        reasons.append(domain_evidence_reason(query))
    if domain_consistency < 0.65 or leakage_risk in {"medium", "high"}:
        reasons.append("Domain consistency sinyali düşük olduğu için semantic leakage riski skoru aşağı çekiyor.")
    else:
        reasons.append("Domain consistency güçlü; öneriler hastalık, modalite ve yöntem bağlamında uyumlu kalıyor.")

    if is_alzheimer_context(query):
        alzheimer_reasons = [
            "Alzheimer tanısı ve erken tespit bağlamı yayın potansiyelini güçlendirir.",
            "EEG/MRI/nörogörüntüleme odağı, ölçülebilir biyomedikal kanıt üretimini destekler.",
            "Açıklanabilir yapay zekâ ve yorumlanabilir model çıktıları klinik doğrulama değerini artırır.",
            "Dış doğrulama ve karşılaştırılabilir veri seti planı SCI düzeyinde ayrışmayı güçlendirir.",
        ]
        reasons = alzheimer_reasons + [
            reason for reason in reasons
            if not any(term in normalize_topic_key(reason) for term in ["wearable", "biosensor", "device reliability"])
        ]

    reasons = [turkishize_report_terms(reason) for reason in reasons[:5]]
    if not reasons:
        reasons = ["Bu değerlendirme, konu kapsamı ve mevcut literatür sinyallerine göre tahmini olarak üretildi."]

    next_action = turkishize_report_terms(domain_narrowing_direction(query, domain_reasoning))

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


def intent_topics_to_dataframe(query: str, selected_domain: str | None = None, score: int = 68) -> pd.DataFrame:
    rows = [
        (item["title"], score, "positive", item["rationale"])
        for item in curated_topic_recommendations(query, max_items=5)
    ]
    out = pd.DataFrame(rows, columns=["suggested_research_topic", "gap_score", "growth_rate", "recommendation"])
    out["domain_consistency_score"] = 0.9
    out["leakage_risk"] = "Low"
    return out


def clean_suggestions_with_curated_bank(suggestions: pd.DataFrame, query: str, min_items: int = 5) -> pd.DataFrame:
    suggestions = _as_dataframe(suggestions)
    topic_col = "suggested_research_topic" if "suggested_research_topic" in suggestions.columns else "suggested_topic" if "suggested_topic" in suggestions.columns else None

    if topic_col:
        clean = suggestions.copy()
        clean = clean[
            ~clean[topic_col].fillna("").astype(str).map(
                lambda title: is_bad_suffix_title(title, query) or is_near_duplicate_title(title, query)
            )
        ].copy()
    else:
        clean = pd.DataFrame()

    curated = intent_topics_to_dataframe(query)
    if clean.empty:
        return curated.head(min_items)

    clean = pd.concat([clean, curated], ignore_index=True, sort=False)
    topic_col = "suggested_research_topic" if "suggested_research_topic" in clean.columns else "suggested_topic" if "suggested_topic" in clean.columns else None
    if topic_col:
        clean = clean.drop_duplicates(subset=[topic_col], keep="first")
        bad_mask = clean[topic_col].fillna("").astype(str).map(
            lambda title: is_bad_suffix_title(title, query) or is_near_duplicate_title(title, query)
        )
        clean = clean.loc[~bad_mask].copy()
    return clean.head(max(min_items, len(clean))).reset_index(drop=True)


def domain_adapted_suggestions(query: str) -> pd.DataFrame:
    selected_field = current_selected_field()
    selected_domain = compatibility_domain(selected_field)
    intent = domain_intent(query, selected_field)
    if selected_domain == ENGINEERING_DOMAIN:
        return intent_topics_to_dataframe(query, selected_field)
    if selected_domain == HEALTHCARE_DOMAIN and intent["subdomain_key"] not in {"general_healthcare", "medicine_clinical"}:
        return intent_topics_to_dataframe(query, selected_field)

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
            (title, 54, "positive", "Sorgudaki biyomedikal odaklar dikkate alınarak öneri oluşturuldu."),
        ]

    out = pd.DataFrame(rows, columns=["suggested_research_topic", "gap_score", "growth_rate", "recommendation"])
    scores = [domain_consistency_score(query, row["suggested_research_topic"])[0] for _, row in out.iterrows()]
    out["domain_consistency_score"] = scores
    out["leakage_risk"] = ["Low" if score >= 0.75 else "Medium" for score in scores]
    return out


def apply_domain_reasoning_filter(suggestions: pd.DataFrame, query: str) -> pd.DataFrame:
    suggestions = _as_dataframe(suggestions)
    fallback = domain_adapted_suggestions(query)
    selected_domain = current_selected_field()

    if suggestions.empty:
        fallback.attrs["domain_filtered_count"] = 0
        return sanitize_suggestions_for_intent(fallback, query, selected_domain)

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
    return sanitize_suggestions_for_intent(filtered, query, selected_domain)


def naturalize_suggestions(suggestions: pd.DataFrame, base_query: str) -> pd.DataFrame:
    if suggestions.empty:
        return intent_topics_to_dataframe(base_query)

    out = suggestions.copy()
    selected_domain = current_selected_field()
    title_col = None

    if "suggested_research_topic" in out.columns:
        title_col = "suggested_research_topic"
    elif "suggested_topic" in out.columns:
        title_col = "suggested_topic"

    if title_col:
        if selected_domain in {ENGINEERING_DOMAIN, HEALTHCARE_DOMAIN}:
            return clean_suggestions_with_curated_bank(sanitize_suggestions_for_intent(out, base_query, selected_domain), base_query)

        def _format(row):
            raw_title = str(row.get(title_col, ""))
            base_topic = row.get("base_topic", "")
            suffix = raw_title.replace(base_query, "", 1).strip().strip('"')
            topic = base_topic or suffix
            return naturalize_topic_title(base_query, topic)

        out[title_col] = out.apply(_format, axis=1)
        out = out.drop_duplicates(subset=[title_col], keep="first").reset_index(drop=True)

    return clean_suggestions_with_curated_bank(out, base_query)


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
            "selected_field": config.get("selected_field", current_selected_field()),
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
        "selected_field": config.get("selected_field", current_selected_field()),
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
                "selected_field": config.get("selected_field", current_selected_field()),
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
        **TURKISH_ENGINEERING_MAP,
        **TURKISH_HEALTHCARE_MAP,
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
    domain_text = migrate_legacy_field(selected_domain or current_selected_field())
    intent = domain_intent(seed, domain_text)
    objects = ", ".join(intent.get("preferred_objects", [])[:4])
    metrics = ", ".join(intent.get("preferred_metrics", [])[:4])
    return (
        "Convert the user's Turkish, English, or messy keywords into 5 concise Q1/SCI-style English research topics. "
        "Avoid generic titles and avoid broad 'AI-based ...' phrasing. "
        "Preserve the user's actual research object/subdomain; AI terms are only methodological modifiers. "
        "Do not append unrelated suffixes such as EEG, COVID-19, Healthcare Security, VANETs, or patient risk prediction unless they are explicit in the user input and aligned with the selected domain. "
        "Each title must include a clear domain object, method, metric/task, and novelty angle when possible. "
        f"Selected supported research field: {domain_text}. Stay strictly within this field. "
        f"Detected subdomain: {intent['subdomain_label']}. Preferred objects: {objects}. Preferred metrics: {metrics}. "
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
    return generate_intent_titles(seed, ENGINEERING_DOMAIN, min_count=5)


def healthcare_topic_refinement(seed: str) -> list[dict[str, str]]:
    return generate_intent_titles(seed, HEALTHCARE_DOMAIN, min_count=5)


def rule_based_topic_refinement(seed: str, selected_domain: str | None = None) -> list[dict[str, str]]:
    return curated_topic_recommendations(seed, max_items=5)

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


def refine_research_topics_legacy(seed: str, provider: str = "", api_key: str = "") -> tuple[list[dict[str, str]], str]:
    if not str(seed or "").strip():
        return [], "No input"
    return curated_topic_recommendations(seed, max_items=5), "Curated biomedical recommendations"


def refine_research_topics(seed: str, provider: str = "", api_key: str = "", selected_domain: str | None = None) -> tuple[list[dict[str, str]], str]:
    if not str(seed or "").strip():
        return [], "No input"
    return curated_topic_recommendations(seed, max_items=5), "Curated biomedical recommendations"


def apply_suggested_research_topic(title: str) -> None:
    clean_title = re.sub(r"\s+", " ", str(title or "")).strip()
    if not clean_title:
        return

    st.session_state["pending_research_topic"] = clean_title
    st.session_state["selected_research_topic"] = clean_title

    for key in QUERY_WIDGET_KEYS.values():
        st.session_state[key] = clean_title

    st.session_state["topic_transfer_notice"] = "Önerilen konu araştırma konusu alanına aktarıldı."


def render_topic_suggester() -> None:
    with st.expander("Araştırma Konusu Öner", expanded=False):
        if "topic_suggester_seed" not in st.session_state:
            st.session_state["topic_suggester_seed"] = ""
        seed = st.text_area(
            "Türkçe veya İngilizce anahtar kelimelerinizi yazın",
            placeholder="biyosensör sinyal kalitesi\nalzheimer mri yapay zeka\nwearable biosensor ECG",
            key="topic_suggester_seed",
            height=90,
        )

        selected_field = current_selected_field()

        if st.button("Konu Öner", use_container_width=True, key="suggest_research_topics_button"):
            st.session_state.pop("topic_suggester_notice", None)
            st.session_state.pop("topic_suggester_notice_type", None)
            st.session_state["topic_suggester_rotation"] = int(st.session_state.get("topic_suggester_rotation", 0)) + 1
            domain_ok, domain_message, domain_debug = validate_domain_query(seed, selected_field)
            st.session_state["domain_guard_debug"] = domain_debug

            if not domain_ok:
                st.session_state["topic_suggestions"] = []
                st.session_state["topic_suggester_results"] = []
                st.session_state["topic_suggester_notice"] = f"{domain_message}\n\n{BIOMEDICAL_KEYWORD_SUGGESTION}"
                st.session_state["topic_suggester_notice_type"] = "error"
            else:
                topics = curated_topic_recommendations(seed, max_items=5)
                st.session_state["topic_suggestions"] = topics
                st.session_state["topic_suggester_results"] = topics
                if domain_message:
                    st.session_state["topic_suggester_notice"] = domain_message
                    st.session_state["topic_suggester_notice_type"] = "warning"

        notice = st.session_state.get("topic_suggester_notice", "")
        notice_type = st.session_state.get("topic_suggester_notice_type", "warning")
        if notice:
            if notice_type == "error":
                st.error(notice)
            else:
                st.warning(notice)

        topics = st.session_state.get("topic_suggestions", st.session_state.get("topic_suggester_results", []))
        if topics:
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
    raw_query = config.get("raw_query", config["query"])
    selected_field = migrate_legacy_field(config.get("selected_field", current_selected_field()))
    selected_domain = compatibility_domain(selected_field)
    retrieval_query = config.get("query", raw_query)
    query = simplify_biomedical_retrieval_query(preprocess_research_query(normalize_keywords_for_domain(retrieval_query)["expanded"]))
    config = config.copy()
    config["query"] = query
    config["selected_domain"] = selected_domain
    config["selected_field"] = selected_field
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
    insufficient_data = df.empty
    if insufficient_data:
        gap_score = {
            "total_records": 0,
            "growth_rate": 0,
            "gap_score": "Yetersiz veri",
            "interpretation": "Hesaplanamadı / güvenilir değil",
            "matching_method": "semantic topic-aware matching",
        }
        warnings.append("Yetersiz veri: Bu sorgu PubMed ve OpenAlex üzerinde yeterli kayıt döndürmedi.")
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
    if insufficient_data:
        strategic_score = "Yetersiz veri"
    ai_insight = generate_ai_insight(
        gap_score,
        top_topics,
        top_keywords,
        query_trend,
        ai_suggestions,
        source_distribution_after,
        query,
    )
    research_strategy = build_research_strategy(query, ai_suggestions, selected_field)
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
        "selected_field": selected_field,
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


def _is_openalex_rate_limit_error(message: str) -> bool:
    key = str(message or "").lower()
    return "429" in key or "too many requests" in key or "rate limit" in key


def _run_openalex_gap(config: dict, warnings: list[str], errors: list[str]) -> dict | None:
    if "_openalex_gap" in config:
        return config["_openalex_gap"]

    openalex_contact = str(config.get("openalex_api_key", "")).strip() or get_config_value("OPENALEX_EMAIL")

    if not openalex_contact:
        warnings.append("OpenAlex e-posta veya API bilgisi eklenmemiş. Yoğun kullanımda erişim sınırlanabilir.")

    last_error = ""
    for attempt, delay in enumerate([0, 1.5, 3, 6]):
        if delay:
            time.sleep(delay)

        try:
            return openalex_gap_analysis(
                query=config["query"],
                api_key=openalex_contact,
                per_page=int(config.get("openalex_max_results", DEFAULT_LIVE_RESULT_LIMIT)),
                years_back=int(config.get("years_back", 5)),
            )
        except Exception as exc:
            last_error = str(exc)
            if _is_openalex_rate_limit_error(last_error) and attempt < 3:
                continue
            break

    if _is_openalex_rate_limit_error(last_error):
        warnings.append("OpenAlex geçici yoğunluk nedeniyle yanıt vermedi. Lütfen birkaç saniye sonra tekrar deneyin.")
    else:
        errors.append("OpenAlex canlı gap analizi şu anda tamamlanamadı. Lütfen birkaç saniye sonra tekrar deneyin.")
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
    return intent_topics_to_dataframe(query, config.get("selected_field", current_selected_field()))

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


def make_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    return df.loc[:, ~df.columns.duplicated()].copy()


def safe_text(value, default: str = "-") -> str:
    try:
        if value is None or pd.isna(value):
            return default
    except Exception:
        if value is None:
            return default

    if isinstance(value, (list, tuple, set)):
        cleaned = []
        for item in value:
            try:
                if item is None or pd.isna(item):
                    continue
            except Exception:
                if item is None:
                    continue
            cleaned.append(str(item))
        return ", ".join(cleaned) if cleaned else default

    return str(value)


def safe_join(values, sep: str = ", ") -> str:
    if values is None:
        return ""

    cleaned = []
    for value in values:
        try:
            if value is None or pd.isna(value):
                continue
        except Exception:
            if value is None:
                continue
        cleaned.append(str(value))
    return sep.join(cleaned)


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
    selected_field = migrate_legacy_field(results.get("selected_field", current_selected_field()))
    selected_domain = compatibility_domain(selected_field)
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

    if selected_domain == ENGINEERING_DOMAIN:
        intent = domain_intent(results.get("query", ""), selected_field)
        objects = ", ".join(intent.get("preferred_objects", [])[:2]) or "the target engineering system"
        metrics = ", ".join(intent.get("preferred_metrics", [])[:2]) or "engineering performance"
        field_openings = {
            "Electrical & Electronics Engineering": "This topic represents a focused AI-compatible electrical and electronics engineering research area.",
            "Biomedical Engineering": "This topic represents a focused biomedical engineering research area.",
        }
        return sanitize_text_for_intent(
            f"{field_openings.get(selected_field, f'This topic represents {competition} and {opportunity_phrase} engineering research area.')} "
            f"The current evidence landscape should be evaluated through {objects}, "
            f"{metrics}, benchmark datasets, and validation evidence, {growth_phrase}. "
            f"Based on the combined opportunity, domain consistency, and paperability signals, the topic shows "
            f"{publication_phrase}, provided the study preserves the real research object and frames AI as a methodological modifier.",
            intent,
        )

    intent = domain_intent(results.get("query", ""), selected_field)
    field_openings = {
        "AI-Compatible Medicine": "This topic represents a focused AI-compatible medical research area.",
        "Nursing": "This topic represents a focused nursing research area.",
        "Midwifery": "This topic represents a focused midwifery research area.",
    }
    summary = (
        f"{field_openings.get(selected_field, f'This topic represents {competition} and {opportunity_phrase} research area within {clinical_domain}.')} "
        f"The current evidence landscape is primarily shaped by {intent.get('preferred_objects', ['the available evidence base'])[0]}, {growth_phrase}. "
        f"Based on the combined opportunity, domain consistency, and paperability signals, the topic shows "
        f"{publication_phrase}, provided the study is framed around a specific method, validation strategy, "
        f"and clinically meaningful differentiation.{pubmed_note}"
    )
    return sanitize_text_for_intent(summary, intent)


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
    query = results.get("raw_query") or results.get("query", "")
    return [item["title"] for item in curated_topic_recommendations(query, max_items=min_count)]

def build_opportunity_analysis(results: dict) -> dict[str, str]:
    gap = results.get("gap_score") or {}
    paperability = results.get("paperability_score") or {}
    matched = parse_numeric(gap.get("total_records"))
    strategic = parse_numeric(results.get("strategic_opportunity_score"))
    paper_score = parse_numeric(paperability.get("total_score"))
    domain = results.get("domain_reasoning") or {}
    selected_field = migrate_legacy_field(results.get("selected_field", current_selected_field()))
    selected_domain = compatibility_domain(selected_field)
    intent = domain_intent(results.get("query", ""), selected_field)

    competition = (
        "The field appears crowded and requires a narrow claim." if matched > 250
        else "The field is competitive but still differentiable with a focused method." if matched > 50
        else "The field has manageable competition and room for positioning."
    )
    if selected_domain == ENGINEERING_DOMAIN:
        novelty = (
            "Novelty is strongest when the topic is framed around a specific engineering system, validation benchmark, and measurable performance contribution."
            if strategic >= 60 else
            "Novelty should be strengthened by narrowing the target infrastructure or engineering system, data source, and validation benchmark."
        )
        relevance = "Engineering relevance is supported by reliability, resilience, optimization, real-time monitoring, or deployment-oriented performance metrics."
        method = "Methodological strength is supported by sensor fusion, simulation, anomaly detection, digital twin modeling, or benchmarked optimization." if paper_score >= 60 else "Methodological strength should be increased through stronger baselines, simulations and reliability analysis."
        dataset = "Dataset feasibility depends on access to sensor streams, simulation outputs, benchmark datasets, system logs, or field measurements."
        return {
            "Competition Level": competition,
            "Novelty Potential": novelty,
            "Engineering Relevance": relevance,
            "Methodological Strength": method,
            "Dataset Feasibility": dataset,
        }

    if selected_domain == HEALTHCARE_DOMAIN and intent["subdomain_key"] not in {"general_healthcare", "medicine_clinical"}:
        label = intent["subdomain_label"]
        objects = ", ".join(intent.get("preferred_objects", [])[:2]) or "the target evidence base"
        metrics = ", ".join(intent.get("preferred_metrics", [])[:2]) or "validated outcomes"
        return {
            "Competition Level": competition,
            "Novelty Potential": f"Novelty is strongest when the topic remains anchored in {label}, {objects}, and measurable {metrics}.",
            "Healthcare Relevance": f"The practical relevance comes from {label.lower()} outcomes rather than unrelated engineering or AI suffixes.",
            "Methodological Strength": healthcare_methodology_for_intent(intent),
            "Dataset Feasibility": evidence_focus_for_intent(intent),
        }

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
    selected_field = migrate_legacy_field(results.get("selected_field", current_selected_field()))
    selected_domain = compatibility_domain(selected_field)
    intent = domain_intent(results.get("query", ""), selected_field)
    score = parse_numeric(paperability.get("total_score"))
    consistency = domain.get("domain_consistency", "Medium")

    if selected_domain == ENGINEERING_DOMAIN:
        if score >= 70 and consistency == "High":
            return (
                "This engineering topic should be pursued with a narrow system boundary, explicit engineering performance metrics, "
                "and benchmarked validation. A focus on sensor fusion, simulation-backed evidence, reliability analysis, or real-time "
                "deployment constraints would improve differentiation compared with generic AI-for-engineering studies."
            )
        if score >= 50:
            return (
                "This engineering topic is worth pursuing after refinement. The strongest path is to define the target infrastructure "
                "or engineering system, select benchmark datasets or simulation scenarios, and position the contribution around measurable "
                "reliability, optimization, resilience, or monitoring gains."
            )
        return (
            "This engineering topic should be reframed before execution. Publication potential depends on clearer system boundaries, "
            "stronger validation evidence, and defensible engineering performance metrics."
        )

    if selected_domain == HEALTHCARE_DOMAIN and intent["subdomain_key"] not in {"general_healthcare", "medicine_clinical"}:
        objects = ", ".join(intent.get("preferred_objects", [])[:2]) or "the target evidence base"
        metrics = ", ".join(intent.get("preferred_metrics", [])[:2]) or "validated outcomes"
        return (
            f"This {intent['subdomain_label']} topic should be pursued with a focused evidence base around {objects}, "
            f"measurable {metrics}, and transparent validation. The strongest publication path is to preserve the healthcare "
            "subdomain intent and avoid unrelated engineering or generic AI framing."
        )

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
        from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except Exception:
        return False

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    font_name = _register_pdf_font()

    def num(value, default=0):
        try:
            return int(float(value))
        except Exception:
            return default

    def pct(value):
        parsed = parse_numeric(value)
        return f"{parsed:.1f}%" if parsed else str(value or "-")

    def clean_text(value: str) -> str:
        return re.sub(r"\s+", " ", str(value or "-")).strip()

    def turkish_gap_comment(gap: dict, strategic_score) -> str:
        matched = parse_numeric(gap.get("total_records"))
        growth = parse_numeric(gap.get("growth_rate"))
        strategic = parse_numeric(strategic_score)
        if safe_text(gap.get("gap_score", "")).lower() == "yetersiz veri" or matched == 0:
            return "Yetersiz veri: Bu sorgu PubMed ve OpenAlex üzerinde güvenilir yorum için yeterli kayıt döndürmedi."
        if strategic >= 70 and growth >= 0:
            return "Bu konu güçlü bir araştırma fırsatı sinyali taşır; ancak yayın potansiyelini artırmak için yöntem, veri türü ve doğrulama planı net biçimde daraltılmalıdır."
        if matched > 50:
            return "Bu konu görünür ve rekabetçi bir literatüre sahiptir. En iyi strateji, geniş başlık yerine daha özgün alt problem ve ölçülebilir katkı tanımlamaktır."
        if matched > 0:
            return "Bu konu için anlamlı fakat yönetilebilir düzeyde literatür bulunmaktadır. Daraltılmış bir araştırma sorusu ile takip edilmeye uygundur."
        return "Bu konu için eşleşme sınırlıdır. Sonuçlar yorumlanırken sorgu kapsamı ve veri kaynağı kapsaması ayrıca kontrol edilmelidir."

    def turkish_final_recommendation(results: dict) -> tuple[str, str, str]:
        gap = results.get("gap_score") or {}
        paperability = results.get("paperability_score") or {}
        df = _as_dataframe(results.get("normalized_dataset"))
        if df.empty or safe_text(gap.get("gap_score", "")).lower() == "yetersiz veri":
            return (
                "Hayır, bu haliyle takip edilmeye uygun değildir.",
                "Sorgu daha sade ve biyomedikal mühendisliği odaklı yeniden yazılmalıdır.",
                "Yetersiz veri nedeniyle yayın potansiyeli güvenilir biçimde değerlendirilemedi.",
            )

        strategic = parse_numeric(results.get("strategic_opportunity_score", gap.get("gap_score")))
        paper_score = parse_numeric(paperability.get("total_score"))
        selected_field = migrate_legacy_field(results.get("selected_field", current_selected_field()))
        display_query = results.get("raw_query") or results.get("query", "")
        if is_alzheimer_context(display_query):
            narrowing = _alzheimer_narrowing_text(display_query)
        else:
            intent = domain_intent(results.get("query", ""), selected_field)
            objects = safe_join(intent.get("preferred_objects", [])[:2]) or "seçilen araştırma nesnesi"
            metrics = safe_join(intent.get("preferred_metrics", [])[:2]) or "ölçülebilir performans çıktıları"
            narrowing = f"Konuyu {objects} odağında; {metrics}, açık doğrulama protokolü ve karşılaştırılabilir performans ölçütleriyle daraltın."
        narrowing = turkishize_report_terms(narrowing)
        pursue = "Evet, takip edilmeli." if strategic >= 45 or paper_score >= 45 else "Önce yeniden daraltılmalı."
        potential = (
            "Yayın potansiyeli güçlü görünüyor; ana risk rekabet düzeyi ve doğrulama kalitesidir."
            if paper_score >= 65 else
            "Yayın potansiyeli orta düzeydedir; katkı iddiası ve veri/kanıt planı güçlendirilmelidir."
            if paper_score >= 40 else
            "Yayın potansiyeli şu haliyle sınırlıdır; konu daha net ve uygulanabilir bir alt probleme indirgenmelidir."
        )
        return pursue, narrowing, potential

    try:
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            rightMargin=1.25 * cm,
            leftMargin=1.25 * cm,
            topMargin=1.15 * cm,
            bottomMargin=1.15 * cm,
            title="ResearchMind AI Biyomedikal Mühendisliği Araştırma Raporu",
        )
        width = A4[0] - doc.leftMargin - doc.rightMargin
        title_style = ParagraphStyle("RmTitle", fontName=font_name, fontSize=22, leading=27, textColor=colors.HexColor("#0f172a"), spaceAfter=4)
        subtitle_style = ParagraphStyle("RmSubtitle", fontName=font_name, fontSize=11, leading=15, textColor=colors.HexColor("#2563eb"), spaceAfter=10)
        h1 = ParagraphStyle("RmH1", fontName=font_name, fontSize=14, leading=18, textColor=colors.HexColor("#12324a"), spaceBefore=8, spaceAfter=6)
        h2 = ParagraphStyle("RmH2", fontName=font_name, fontSize=9, leading=12, textColor=colors.HexColor("#2563eb"))
        body = ParagraphStyle("RmBody", fontName=font_name, fontSize=8.7, leading=12.2, textColor=colors.HexColor("#243447"))
        small = ParagraphStyle("RmSmall", fontName=font_name, fontSize=8, leading=11, textColor=colors.HexColor("#475569"))

        def p(text, style=body):
            clean = html.escape(clean_text(safe_text(text))).replace("&lt;br/&gt;", "<br/>")
            return Paragraph(clean, style)

        def section(title_text):
            return [Spacer(1, 0.12 * cm), p(title_text, h1), HRFlowable(width="100%", color=colors.HexColor("#dbeafe")), Spacer(1, 0.08 * cm)]

        def simple_table(rows, col_widths=None, header=False, bg="#f8fafc"):
            safe_rows = [[safe_text(cell) for cell in row] for row in (rows or [])]
            if not safe_rows:
                safe_rows = [["-"]]
            data = [[p(cell, h2 if header and r == 0 else body) for cell in row] for r, row in enumerate(safe_rows)]
            style = [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(bg)),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#dbeafe")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
            if header:
                style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0f2fe")))
            col_count = max(1, len(safe_rows[0]))
            return Table(data, colWidths=col_widths or [width / col_count] * col_count, style=TableStyle(style))

        df = _as_dataframe(results.get("normalized_dataset"))
        gap = results.get("gap_score") or {}
        openalex_gap = results.get("openalex_gap") or {}
        suggestions = clean_suggestions_with_curated_bank(
            _as_dataframe(results.get("ai_topic_suggestions")),
            results.get("query", ""),
        )
        semantic = _as_dataframe(results.get("semantic_search_results"))
        distribution = results.get("source_distribution", {}) or {}
        selected_field = migrate_legacy_field(results.get("selected_field", current_selected_field()))
        strategic_score = results.get("strategic_opportunity_score", gap.get("gap_score", "-"))
        pubmed_count = distribution.get("PubMed", 0)
        openalex_count = distribution.get("OpenAlex", 0)
        local_count = distribution.get("Yerel", 0)
        openalex_estimated_volume = openalex_gap.get("total_records", openalex_count)

        story = [
            p("ResearchMind AI Biyomedikal Mühendisliği Araştırma Raporu", title_style),
            p("OpenAlex Destekli Research Gap ve Yayın Potansiyeli Analizi", subtitle_style),
            HRFlowable(width="100%", color=colors.HexColor("#38bdf8")),
        ]

        story += section("1. Genel Bilgiler")
        story.append(simple_table([
            ["Araştırma Alanı", "Biyomedikal Mühendisliği"],
            ["Araştırma Konusu", results.get("query", "-")],
            ["Veri Kaynağı", results.get("data_source_label", results.get("data_source", "-"))],
            ["Analiz Tarihi", results.get("analysis_time", "-")],
            ["Filtreleme Sonrası Analiz Edilen Yayın Sayısı", f"{len(df):,}"],
            ["PubMed Kayıt Sayısı", str(pubmed_count)],
            ["Analize Dahil Edilen OpenAlex Kayıt Sayısı", str(openalex_count)],
            ["OpenAlex Tahmini Toplam Yayın Hacmi", str(openalex_estimated_volume)],
        ], col_widths=[width * 0.35, width * 0.65]))
        story.append(Spacer(1, 0.08 * cm))
        story.append(p("OpenAlex tahmini toplam yayın hacmi, sorgunun genel akademik görünürlüğünü temsil eder. Analize dahil edilen kayıtlar ise filtreleme ve işleme sonrası kullanılan örnek yayın kümesini ifade eder.", small))
        if pubmed_count == 0 and openalex_count > 0:
            story.append(Spacer(1, 0.08 * cm))
            story.append(p("PubMed bağlantısı başarılı ancak bu spesifik sorgu için sonuç bulunamadı. Analiz OpenAlex verileriyle tamamlandı.", small))

        story += section("2. Konu Trendi ve Research Gap Skoru")
        story.append(simple_table([
            ["Eşleşen kayıt sayısı", str(gap.get("total_records", "-"))],
            ["Büyüme oranı", str(gap.get("growth_rate", "-"))],
            ["Research Gap Skoru", str(gap.get("gap_score", "-"))],
            ["Stratejik Fırsat Skoru", str(strategic_score)],
            ["Türkçe Değerlendirme", turkish_gap_comment(gap, strategic_score)],
        ], col_widths=[width * 0.35, width * 0.65], bg="#f0f9ff"))

        story += section("3. Benzer Akademik Çalışmalar")
        sim_rows = [["Başlık", "Dergi", "Yıl", "Yayın türü", "Benzerlik"]]
        for _, row in semantic.head(10).iterrows():
            sim = row.get("similarity_score", "")
            sim_text = f"{parse_numeric(sim) * 100:.1f}%" if sim != "" else "-"
            sim_rows.append([
                row.get("title", "-"),
                row.get("journal", "-"),
                row.get("pub_year", row.get("year", row.get("publication_year", "-"))),
                row.get("research_type", "-"),
                sim_text,
            ])
        if len(sim_rows) == 1:
            sim_rows.append(["Benzer çalışma bulunamadı", "-", "-", "-", "-"])
        story.append(simple_table(sim_rows, col_widths=[width * 0.42, width * 0.18, width * 0.10, width * 0.18, width * 0.12], header=True))

        story += section("4. OpenAlex Canlı Gap Analizi")
        story.append(p("OpenAlex sonuçları, konunun genel akademik görünürlüğünü ve son yıllardaki yayın hacmini değerlendirmek için kullanılmıştır.", body))
        story.append(simple_table([
            ["OpenAlex Tahmini Toplam Yayın Hacmi", str(openalex_estimated_volume)],
            ["OpenAlex Büyüme Oranı", str(openalex_gap.get("growth_rate", "-"))],
            ["OpenAlex Research Gap Skoru", str(openalex_gap.get("gap_score", "-"))],
        ], col_widths=[width * 0.40, width * 0.60], bg="#ecfeff"))
        trend = _as_dataframe(openalex_gap.get("trend"))
        if not trend.empty:
            trend_rows = [["Yıl", "Yayın sayısı"]]
            year_col = "publication_year" if "publication_year" in trend.columns else "period" if "period" in trend.columns else None
            count_col = "publication_count" if "publication_count" in trend.columns else "count" if "count" in trend.columns else None
            if year_col and count_col:
                for _, row in trend.tail(8).iterrows():
                    trend_rows.append([row.get(year_col, "-"), row.get(count_col, "-")])
                story.append(Spacer(1, 0.08 * cm))
                story.append(simple_table(trend_rows, col_widths=[width * 0.35, width * 0.65], header=True))

        story += section("5. Araştırma Konusu Önerileri")
        if suggestions.empty:
            suggestions = intent_topics_to_dataframe(results.get("query", ""), selected_field)
        if df.empty:
            story.append(p("Sorguyu iyileştirmek için önerilen biyomedikal başlıklar aşağıda listelenmiştir.", small))
        topic_col = "suggested_research_topic" if "suggested_research_topic" in suggestions.columns else "suggested_topic" if "suggested_topic" in suggestions.columns else None
        suggestion_rows = [["Başlık", "Fırsat Düzeyi", "Gap Skoru", "Kısa Gerekçe"]]
        if topic_col:
            for _, row in suggestions.head(5).iterrows():
                trend_status, _ = opportunity_trend_status(row.get("gap_score", "-"), row.get("growth_rate", "-"))
                suggestion_rows.append([
                    row.get(topic_col, "-"),
                    trend_status,
                    row.get("gap_score", "-"),
                    localize_text(row.get("recommendation", "")) or "Seçilen araştırma alanı ile uyumlu öneri.",
                ])
        if len(suggestion_rows) == 1:
            suggestion_rows.append(["Bu analiz için otomatik öneri üretilemedi. Konunun daha net anahtar kelimelerle yeniden çalıştırılması önerilir.", "-", "-", "-"])
        story.append(simple_table(suggestion_rows, col_widths=[width * 0.42, width * 0.18, width * 0.12, width * 0.28], header=True, bg="#fff7ed"))

        story += section("6. Sonuç ve Stratejik Öneri")
        pursue, narrowing, potential = turkish_final_recommendation(results)
        story.append(simple_table([
            ["Konu takip edilmeli mi?", pursue],
            ["Daraltma önerisi", narrowing],
            ["Yayın potansiyeli yorumu", potential],
            ["Not", "Bu rapor karar destek amaçlıdır; yayın garantisi anlamına gelmez."],
        ], col_widths=[width * 0.35, width * 0.65], bg="#eef2ff"))

        doc.build(story)
        return output_path.exists()
    except Exception:
        return False

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
    selected_field = migrate_legacy_field(results.get("selected_field", current_selected_field()))
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
        f"Araştırma alanı: Biyomedikal Mühendisliği",
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
        f"- Tried fallback queries: {safe_join(diagnostics.get('pubmed_fallback_queries') or []) or '-'}",
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
        safe_text(results.get("ai_research_insight", "Insight üretilemedi.")),
        "",
        "Research Strategy Engine:",
        f"- Suggested Research Direction: {results.get('research_strategy', {}).get('direction', '-')}",
        f"- Recommended Methodology: {results.get('research_strategy', {}).get('methodology', '-')}",
        f"- Suggested Dataset / Evidence Focus: {results.get('research_strategy', {}).get('evidence', '-')}",
        f"- Differentiation Strategy: {results.get('research_strategy', {}).get('differentiation', '-')}",
        "",
        "Field Reasoning:",
        f"- Selected research field: {selected_field}",
        f"- Detected field: {domain_reasoning.get('subdomain_label', domain_reasoning.get('primary_domain', '-'))}",
        f"- Field confidence: {domain_reasoning.get('subdomain_confidence', '-')}",
        f"- Preferred objects: {domain_reasoning.get('preferred_objects', '-')}",
        f"- Preferred metrics: {domain_reasoning.get('preferred_metrics', '-')}",
        f"- Primary disease: {domain_reasoning.get('primary_disease', '-')}",
        f"- Modality: {domain_reasoning.get('primary_modality', '-')}",
        f"- Clinical domain: {domain_reasoning.get('clinical_domain', '-')}",
        f"- Dominant methodology: {domain_reasoning.get('primary_method', '-')}",
        f"- Field consistency: {domain_reasoning.get('domain_consistency', '-')} ({domain_reasoning.get('domain_consistency_score', '-')})",
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
        logo_src = logo_data_uri()
        if logo_src:
            st.markdown(
                f"""
                <div style="text-align:center; margin: 0.2rem 0 0.9rem;">
                    <div style="background: rgba(255,255,255,0.94); padding: 10px 14px; border-radius: 14px; display: inline-block; box-shadow: 0 10px 26px rgba(15,23,42,0.16);">
                        <img src="{logo_src}" style="width:210px; max-width:100%; height:auto; display:block;" />
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown(
            """
            <div style="text-align:center; margin: 0.25rem 0 1.25rem;">
                <div style="font-weight:700; font-size:1.05rem;">ResearchMind AI</div>
                <div style="font-size:0.78rem; opacity:0.75;">Biomedical Engineering Research Intelligence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.header("Veri Kaynağı ve Analiz Ayarları")
        st.markdown("**Desteklenen Araştırma Alanı**")
        st.markdown("Biomedical Engineering")
        selected_field = BIOMEDICAL_FIELD
        selected_domain = compatibility_domain(selected_field)
        st.session_state["selected_research_domain"] = selected_domain
        st.caption(
            "ResearchMind AI demo sürümü şu anda yalnızca Biyomedikal Mühendisliği alanı için optimize edilmiştir. "
            "Desteklenmeyen alanlar analiz kapsamı dışında bırakılır."
        )
        source_label = st.selectbox(
            "Veri kaynağı",
            list(DATA_SOURCE_LABELS.values()),
            index=list(DATA_SOURCE_LABELS.values()).index("Hibrit Analiz"),
            key="sidebar_data_source",
        )
        data_source = next(
            key for key, value in DATA_SOURCE_LABELS.items() if value == source_label
        )
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
            "selected_field": selected_field,
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

        raw_query = str(config.get("query", "") or "")
        config["raw_query"] = raw_query
        normalized_query = normalize_field_keywords(preprocess_research_query(raw_query))
        if normalized_query:
            config["query"] = simplify_biomedical_retrieval_query(normalized_query)

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
    else:
        st.info("PDF raporu henüz oluşturulmadı.")


def render_results(results: dict) -> None:
    df = _as_dataframe(results.get("normalized_dataset"))
    diagnostics = results.get("diagnostics", {})
    pubmed_user_message = diagnostics.get("pubmed_user_message", "")
    distribution = results.get("source_distribution", {})

    for warning in results.get("warnings", []):
        if "PubMed" in str(warning) and distribution.get("OpenAlex", 0) > 0:
            continue
        st.warning(warning)

    for error in results.get("errors", []):
        if error == pubmed_user_message:
            continue
        st.error(error)

    _render_product_success(results)

    if distribution.get("PubMed", 0) == 0 and distribution.get("OpenAlex", 0) > 0:
        if diagnostics.get("pubmed_error"):
            st.warning("PubMed bağlantı hatası oluştu. Analiz OpenAlex verileriyle tamamlandı.")
        else:
            st.info("PubMed bağlantısı başarılı ancak bu spesifik sorgu için sonuç bulunamadı. Analiz OpenAlex verileriyle tamamlandı.")

    with st.expander("Teknik detayları göster", expanded=False):
        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Analize dahil edilen toplam kayıt", f"{len(df):,}")
        t2.metric("Analize dahil edilen OpenAlex kayıtları", f"{distribution.get('OpenAlex', 0):,}")
        t3.metric("PubMed", f"{distribution.get('PubMed', 0):,}")
        t4.metric("Yerel", f"{distribution.get('Yerel', 0):,}")
        _render_pubmed_diagnostics(results)
        if (not demo_mode_enabled()) or is_admin():
            st.caption(f"Dedup öncesi: {results.get('dedup_before_count', len(df))} | Dedup sonrası: {results.get('dedup_after_count', len(df))}")

    st.divider()
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
                "PubMed ba\u011flant\u0131s\u0131 ba\u015far\u0131l\u0131 ancak bu spesifik sorgu i\u00e7in sonu\u00e7 bulunamad\u0131.",
            )
        )
        return

    if diagnostics.get("pubmed_error"):
        st.warning("PubMed bağlantı hatası oluştu. Analiz OpenAlex verileriyle tamamlandı.")
        return

    st.info(
        "PubMed ba\u011flant\u0131s\u0131 ba\u015far\u0131l\u0131: "
        f"{diagnostics.get('pubmed_pmids_fetched', 0)} PMID \u00e7ekildi, "
        f"{diagnostics.get('pubmed_normalized_records', 0)} kay\u0131t normalize edildi."
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
        title="Yıllara Göre Yayın Trendi",
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
    st.subheader("1) Konu Trendi ve Research Gap Skoru")
    gap = results.get("gap_score") or {}
    qtrend = _as_dataframe(results.get("query_trend"))
    strategic_score = results.get("strategic_opportunity_score", gap.get("gap_score", "-"))
    insufficient = safe_text(strategic_score).lower() == "yetersiz veri" or safe_text(gap.get("gap_score", "")).lower() == "yetersiz veri"
    if insufficient:
        opportunity_label, opportunity_level = "Yetersiz veri", "low"
        big_badge, big_badge_level = "Hesaplanamadı / güvenilir değil", "low"
    else:
        opportunity_label, opportunity_level = strategic_level(strategic_score)
        big_badge, big_badge_level = opportunity_status(strategic_score)

    g1, g2, g3, g4 = st.columns(4)
    with g1:
        render_metric_card("Eşleşen kayıt", gap.get("total_records", 0), "Analize dahil olan ilgili yayın")
    with g2:
        render_metric_card("Büyüme oranı", gap.get("growth_rate", "-"), "Son dönem eğilim sinyali")
    with g3:
        render_metric_card("Stratejik Fırsat Skoru", strategic_score, f"Research Gap Skoru: {gap.get('gap_score', '-')}", big_badge, big_badge_level)
    with g4:
        render_metric_card("Fırsat seviyesi", opportunity_label, "Skora göre stratejik seviye", big_badge, opportunity_level)

    interpretation = localize_text(gap.get("interpretation", ""))
    if interpretation:
        st.caption(interpretation)
    if insufficient:
        st.warning("Yetersiz veri: Bu sorgu PubMed ve OpenAlex üzerinde yeterli kayıt döndürmedi.")

    if qtrend.empty:
        st.info("Bu araştırma konusu için zamansal trend bulunamadı.")
        return

    fig = px.line(
        qtrend,
        x="period",
        y="publication_count",
        markers=True,
        title="Yıllara Göre Yayın Trendi",
        labels={"period": "Yıl", "publication_count": "Yayın Sayısı"},
    )
    fig = style_plotly_chart(fig)
    fig.update_layout(height=340, margin={"l": 18, "r": 18, "t": 48, "b": 28}, xaxis_title="Yıl", yaxis_title="Yayın Sayısı")
    st.plotly_chart(fig, use_container_width=True)
    trend_badge, trend_level = opportunity_status(strategic_score)
    st.markdown(
        f'<span class="rm-status rm-status-{trend_level}">{trend_badge}</span>',
        unsafe_allow_html=True,
    )


def _render_semantic_results(results: dict) -> None:
    st.subheader("2) Benzer Akademik Çalışmalar")
    st.caption("Girilen araştırma konusuna semantik olarak en yakın çalışmalar listelenir.")
    semantic = _as_dataframe(results.get("semantic_search_results"))

    if semantic.empty:
        st.info("Benzer akademik çalışma sonucu üretilemedi.")
        return

    display = semantic.copy().head(10)
    if "similarity_score" in display.columns:
        display["Benzerlik"] = (
            pd.to_numeric(display["similarity_score"], errors="coerce")
            .fillna(0)
            .map(lambda value: f"{value * 100:.1f}%")
        )
    year_col = "pub_year" if "pub_year" in display.columns else "year" if "year" in display.columns else "publication_year" if "publication_year" in display.columns else None
    columns = {
        "title": "Başlık",
        "journal": "Dergi",
        "research_type": "Yayın türü",
    }
    if year_col:
        columns[year_col] = "Yıl"
    keep = [col for col in ["title", "journal", year_col, "research_type", "Benzerlik"] if col and col in display.columns]
    display = make_unique_columns(display[keep].rename(columns=columns))
    st.dataframe(display, use_container_width=True, hide_index=True)

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

    st.markdown("### Field Reasoning Summary")
    cols = st.columns(5)
    values = [
        ("Primary research field", reasoning.get("subdomain_label", reasoning.get("clinical_domain", "-")), "Selected field"),
        ("Primary modality", reasoning.get("primary_modality", "-"), "Evidence type"),
        ("Dominant methodology", reasoning.get("primary_method", "-"), "Method family"),
        ("Field consistency", consistency, f"Score: {reasoning.get('domain_consistency_score', '-')}"),
        ("Semantic leakage risk", risk, f"Filtered: {reasoning.get('leakage_filtered_count', 0)}"),
    ]

    for index, (label, value, note) in enumerate(values):
        with cols[index]:
            badge = value if label in {"Field consistency", "Semantic leakage risk"} else ""
            badge_level = level if label == "Field consistency" else risk_level
            render_metric_card(label, value, note, badge if label in {"Field consistency", "Semantic leakage risk"} else "", badge_level)

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
    distribution = results.get("source_distribution", {})

    if not openalex_gap and distribution.get("OpenAlex", 0) <= 0:
        return

    st.subheader("3) OpenAlex Canlı Gap Analizi")
    st.caption("OpenAlex sonuçları, konunun genel akademik görünürlüğünü ve son yıllardaki yayın hacmini değerlendirmek için kullanılmıştır.")

    l1, l2, l3 = st.columns(3)
    l1.metric("OpenAlex tahmini toplam yayın hacmi", (openalex_gap or {}).get("total_records", distribution.get("OpenAlex", 0)))
    l1.caption("Bu değer OpenAlex üzerinde sorguyla ilişkili genel akademik görünürlüğü temsil eder.")
    l2.metric("OpenAlex büyüme oranı", (openalex_gap or {}).get("growth_rate", "-"))
    l3.metric("OpenAlex Research Gap Skoru", (openalex_gap or {}).get("gap_score", "-"))

    interpretation = localize_text((openalex_gap or {}).get("interpretation", ""))
    if interpretation:
        st.write(interpretation)

    trend = _as_dataframe((openalex_gap or {}).get("trend"))
    if not trend.empty:
        year_col = "publication_year" if "publication_year" in trend.columns else "period" if "period" in trend.columns else None
        count_col = "publication_count" if "publication_count" in trend.columns else "count" if "count" in trend.columns else None
        if not year_col or not count_col:
            return
        trend_display = pd.DataFrame({
            "Yıl": trend[year_col],
            "Yayın sayısı": trend[count_col],
        })
        trend_display = make_unique_columns(trend_display)
        st.dataframe(trend_display, use_container_width=True, hide_index=True)

def _render_product_success(results: dict) -> None:
    st.success("✅ Analiz başarıyla tamamlandı. Sonuçlar otomatik olarak kaydedildi.")
    if (not demo_mode_enabled()) or is_admin():
        with st.expander("Export klasörünü göster"):
            st.code(results.get("export_path", "-"))

    export_path = results.get("export_path", "")
    export_dir = Path(export_path) if export_path else Path()
    executive_matches = sorted(export_dir.glob("ResearchMind_AI_Executive_Report_*.pdf")) if export_path else []
    executive_pdf_path = executive_matches[-1] if executive_matches else export_dir / "ResearchMind_AI_Executive_Report.pdf"
    if export_path and executive_pdf_path.exists():
        st.download_button(
            label="Executive PDF İndir",
            data=executive_pdf_path.read_bytes(),
            file_name=executive_pdf_path.name,
            mime="application/pdf",
            use_container_width=True,
            key="download_executive_pdf_report_main",
        )
    else:
        st.info("PDF raporu henüz oluşturulmadı.")



def _render_ai_suggestions(results: dict) -> None:
    st.subheader("4) Araştırma Konusu Önerileri")
    suggestions = clean_suggestions_with_curated_bank(
        _as_dataframe(results.get("ai_topic_suggestions")),
        results.get("query", ""),
    )

    if suggestions.empty:
        suggestions = intent_topics_to_dataframe(
            results.get("query", ""),
            results.get("selected_field", current_selected_field()),
        )

    if suggestions.empty:
        st.info("Alan uyumlu Öneri üretilemedi.")
        return

    st.markdown("### Öne Çıkan Araştırma Başlıkları")
    top_items = suggestions.head(5)
    for index, (_, row) in enumerate(top_items.iterrows(), start=1):
        title = row.get("suggested_research_topic", row.get("suggested_topic", "Araştırma önerisi"))
        score = row.get("gap_score", "-")
        growth = row.get("growth_rate", "-")
        recommendation = localize_text(row.get("recommendation", "")) or "Bu başlık, seçilen araştırma alanı ile uyumlu olacak şekilde önerilmiştir."
        trend_status, level = opportunity_trend_status(score, growth)
        st.markdown(
            f"""
            <div class="rm-opportunity">
                <h4>{index}. {title}</h4>
                <span class="rm-status rm-status-{level}">{trend_status}</span>
                <div class="rm-muted" style="margin-top: 0.85rem;">Gap Score</div>
                <div class="rm-card-value">{score}</div>
                <p class="rm-card-note">{recommendation}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    display = suggestions.copy()
    if "recommendation" in display.columns:
        display["Kısa gerekçe"] = display["recommendation"].map(localize_text)
    topic_col = "suggested_research_topic" if "suggested_research_topic" in display.columns else "suggested_topic" if "suggested_topic" in display.columns else None
    if topic_col:
        display["Başlık"] = display[topic_col]
    if "gap_score" in display.columns:
        display["Gap Skoru"] = display["gap_score"]
    keep = [col for col in ["Başlık", "Gap Skoru", "Kısa gerekçe"] if col in display.columns]
    if keep:
        with st.expander("Detaylı öneri tablosunu göster", expanded=False):
            st.dataframe(make_unique_columns(display[keep].head(8)), use_container_width=True, hide_index=True)

inject_product_styles()

if not render_demo_registration_gate():
    st.stop()

sidebar_config = build_sidebar_config()

if sidebar_config["run_clicked"]:
    is_demo = demo_mode_enabled()
    demo_email = st.session_state.get("demo_user_email", "")
    domain_ok, domain_message, domain_debug = validate_domain_query(
        sidebar_config.get("raw_query", sidebar_config.get("query", "")),
        sidebar_config.get("selected_field", current_selected_field()),
    )
    st.session_state["domain_guard_debug"] = domain_debug

    if not domain_ok:
        st.error(f"{domain_message}\n\n{BIOMEDICAL_KEYWORD_SUGGESTION}")
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
    st.info("Araştırma konusunu gir, analiz dönemini seç ve tek tıkla trend, fırsat ve Research Gap Score sonuçlarını üret. Bu demo sürüm yalnızca Biyomedikal Mühendisliği alanı için optimize edilmiştir.")
