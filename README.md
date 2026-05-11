# Temporal Biomedical Trend Analysis System

Bu proje, `biomedical_research_abstracts_2024_2026.csv` veri seti üzerinden 2024-2026 dönemindeki biyomedikal araştırma eğilimlerini analiz eden bir Streamlit tabanlı karar destek prototipidir.

## Proje Başlığı
**Temporal Biomedical Trend Analysis: AI-Based Mapping of Biomedical Research Topics from PubMed Abstracts (2024-2026)**

## Temel Özellikler
- Yıl/ay bazlı yayın trend analizi
- Ülke, dergi, araştırma tipi ve açık erişim dağılımları
- MeSH/keyword/major topic temelli konu analizi
- Anahtar kelime arama ve abstract filtreleme
- Basit research-gap skoru
- CSV kaynaklı tekrarlanabilir analiz pipeline'ı

## Kurulum
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Veri Dosyası
CSV dosyasını proje klasörüne koyun veya Streamlit arayüzünden dosya yolunu girin:

```text
biomedical_research_abstracts_2024_2026.csv
```

## Dosyalar
- `app.py`: Streamlit dashboard
- `trend_engine.py`: veri okuma, aggregation ve trend fonksiyonları
- `requirements.txt`: gerekli paketler
- `project_text.md`: proje pazarına koyulabilecek akademik proje metni
