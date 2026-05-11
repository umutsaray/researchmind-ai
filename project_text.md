# Proje Başlığı
**Temporal Biomedical Trend Analysis: PubMed Tabanlı Biyomedikal Araştırma Eğilimlerinin Yapay Zekâ Destekli Analizi (2024–2026)**

## Kısa Proje Özeti
Bu proje, 2024–2026 dönemine ait PubMed kaynaklı 126.832 temizlenmiş biyomedikal araştırma özeti üzerinden biyomedikal bilimlerde konu eğilimlerini, ülke/dergi dağılımlarını, açık erişim durumlarını ve araştırma türlerini analiz eden yapay zekâ destekli bir karar destek platformu geliştirmeyi amaçlamaktadır. Sistem; abstract metinleri, MeSH/anahtar kelimeler, yayın yılı/ayı, dergi adı, ülke bilgisi, araştırma türü ve açık erişim değişkenlerini kullanarak literatürde yükselen alanları, doygunlaşmış konuları ve potansiyel araştırma boşluklarını belirlemektedir.

## Problem Tanımı
Biyomedikal literatür çok hızlı büyümektedir. Araştırmacılar, hangi konuların yükseldiğini, hangi alanların doygunlaştığını ve hangi alt alanlarda araştırma boşluğu bulunduğunu manuel olarak takip etmekte zorlanmaktadır. Geleneksel literatür arama sistemleri makale bulmaya odaklanırken, zamansal eğilim, ülke bazlı üretkenlik, konu yoğunluğu ve araştırma fırsatı analizi gibi karar destek çıktıları sınırlı kalmaktadır.

## Amaç
Projenin amacı, PubMed abstract verilerini kullanarak biyomedikal araştırma alanlarında zamansal trendleri ve araştırma boşluklarını çıkaran, semantic search ve NLP tabanlı bir analiz platformu geliştirmektir.

## Yöntem
1. Veri ön işleme: abstract, başlık, MeSH/keyword, ülke, dergi ve tarih alanlarının temizlenmesi.
2. Tanımlayıcı analiz: yıl/ay, ülke, dergi, araştırma tipi ve açık erişim dağılımlarının çıkarılması.
3. NLP tabanlı konu analizi: keyword/MeSH frekans analizi, TF-IDF ve ileri aşamada SciBERT/Sentence-BERT embeddingleri.
4. Zamansal trend analizi: seçilen konu veya anahtar kelimenin aylık/yıllık yayın yoğunluğunun hesaplanması.
5. Research-gap skoru: düşük toplam yayın hacmi, son dönem büyüme oranı ve konu doygunluğu göstergeleriyle araştırma fırsatı skoru oluşturulması.
6. Dashboard: araştırmacının konu araması yapabildiği, trend/gap çıktısı alabildiği etkileşimli Streamlit arayüzü.

## Yenilikçi Yön
Proje yalnızca makale arayan bir sistem değildir; biyomedikal literatürdeki konuların zaman içindeki değişimini analiz ederek araştırmacıya stratejik konu seçimi ve araştırma boşluğu önerisi sunar. Bu yönüyle klasik literatür tarama araçlarından ayrılır.

## Beklenen Çıktılar
- Biyomedikal konu trend haritası
- Ülke ve dergi bazlı araştırma üretkenliği analizi
- Açık erişim ve kapalı erişim yayın karşılaştırması
- Araştırma boşluğu skorlama modeli
- Streamlit tabanlı çalışan demo platformu
- Akademik makale ve proje pazarı sunumu için görsel çıktılar

## Ticarileşme Potansiyeli
Sistem; akademisyenler, doktora öğrencileri, Ar-Ge merkezleri, üniversiteler ve bilimsel yayın stratejisi geliştiren kurumlar için literatür istihbaratı platformu olarak sunulabilir. Premium sürümde gelişmiş semantic search, otomatik literatür özeti, ülke/kurum bazlı analiz ve araştırma önerisi modülleri yer alabilir.
