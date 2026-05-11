# ResearchMind AI

ResearchMind AI is a Streamlit-based research intelligence demo platform. It analyzes a research topic with local CSV, OpenAlex, PubMed, or Hybrid live data and produces Research Gap Score, Domain Reasoning, Paperability Score, topic suggestions, and an Executive PDF report.

## Features

- Local CSV, OpenAlex Live, PubMed Live, and Hybrid analysis
- Semantic and domain-aware matching
- Research Gap Score and Strategic Opportunity Score
- Paperability Score / SCI Publication Potential
- Optional LLM-assisted topic refinement with OpenAI, Groq, or Gemini
- Public demo mode with registration, daily usage limit, admin access, logs, and cache
- Executive PDF export

## Local Development

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create a local `.env` file from the example:

```bash
copy .env.example .env
```

3. Fill only the keys you need:

```env
DEMO_MODE=false
DEMO_ACCESS_ENABLED=true
OPENALEX_EMAIL=your-email@example.com
NCBI_EMAIL=your-email@example.com
PUBMED_API_KEY=
GROQ_API_KEY=
OPENAI_API_KEY=
GEMINI_API_KEY=
ADMIN_EMAILS=admin@example.com
ADMIN_PASSWORD_HASH=
ADMIN_BYPASS=false
```

4. Run the app:

```bash
streamlit run app.py
```

## Streamlit Cloud Deployment

In Streamlit Cloud, add the same values under **App settings > Secrets**:

```toml
DEMO_MODE = "true"
DEMO_ACCESS_ENABLED = "true"
OPENALEX_EMAIL = "your-email@example.com"
NCBI_EMAIL = "your-email@example.com"
PUBMED_API_KEY = ""
GROQ_API_KEY = ""
OPENAI_API_KEY = ""
GEMINI_API_KEY = ""
ADMIN_EMAILS = "admin@example.com"
ADMIN_PASSWORD_HASH = ""
ADMIN_BYPASS = "false"
```

The app reads configuration in this order:

1. Streamlit `st.secrets`
2. Environment variables
3. Local `.env`
4. Safe default

## Admin Password Hash

Do not store the admin password as plain text. Generate a SHA-256 hash:

```bash
python -c "import hashlib; print(hashlib.sha256('SENIN_ADMIN_SIFREN'.encode()).hexdigest())"
```

Then set:

```env
ADMIN_PASSWORD_HASH=generated_hash_here
```

For a controlled internal demo, you can temporarily set:

```env
ADMIN_BYPASS=true
```

When enabled, emails listed in `ADMIN_EMAILS` become admin without password.

## Security Notes

- Do not commit `.env` or `.streamlit/secrets.toml`.
- `.gitignore` excludes local secrets, demo logs, demo cache, outputs, and pickle files.
- API keys, emails, phone numbers, and admin passwords are not written to PDF reports.
- Public demo logs are stored locally under `demo_logs/` and are not exposed as downloads.

## Main Files

- `app.py`: Streamlit UI, orchestration, demo mode, exports
- `trend_engine.py`: trend, OpenAlex, scoring, and local analysis helpers
- `pubmed_client.py`: PubMed E-utilities client
- `config_utils.py`: hybrid local `.env` / Streamlit secrets configuration helper
- `requirements.txt`: dependencies
