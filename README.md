# Multi-Profile Research Agent

A multi-agent research and content creation system built with Python, the Anthropic Claude API, and Streamlit. Enter a topic, and three specialized AI agents — Research, Writer, and Editor — work in sequence to produce polished, publication-ready content.

Built by Jim Spellos as both a practical content workflow tool and a live classroom demonstration of how agentic AI systems actually work.

---

## What It Does

1. **Research Agent** — Searches the web, reads YouTube transcripts, scrapes web pages, and digests uploaded files to build comprehensive research notes on your topic.
2. **Writer Agent** — Takes the research notes and produces formatted content based on the selected profile (Patreon article, LinkedIn post, venue search report, etc.).
3. **Editor Agent** *(optional, profile-dependent)* — Fact-checks the written content against the original research notes and polishes the final output.

Output is saved as both a **Word document (.docx)** and a **Markdown file (.md)**, each containing:
- The initial prompt (topic + sources used)
- Full research notes
- Final written/edited content

### Key Features
- **YAML-configured profiles** — swap between content types (Patreon, LinkedIn, venue search, etc.) without touching code
- **Streaming output** — watch each agent work in real time, ideal for live demos
- **Multiple source types** — web search, YouTube transcripts, website scraping (via Firecrawl), and file uploads (PDF, TXT, MD, CSV)
- **Token usage tracking** — optional display of input/output tokens and estimated cost per run
- **"New Research" button** — clears results without losing the page, no accidental screen wipes on download

---

## File Version History

The numbered file prefix reflects the development progression. Each version is a complete, working file — not a patch.

| File | What It Added |
|------|--------------|
| `1_content_agent.py` | Basic Claude API call with system prompt, no search |
| `2_content_agent.py` | Web search tool + markdown file output |
| `3_` – `6_` | Streaming, multi-turn conversation, tool use refinements |
| `7_editor_agent.py` | Three-agent pipeline (Research → Writer → Editor) |
| `8_multi_content_research.py` | YAML profiles, Streamlit UI, Firecrawl, token tracking, cloud deployment |
| `9_multi_content_research.py` | Saves initial prompt in output files; research notes included in Word doc; download buttons persist (no screen clear); New Research button |

**Current production file: `9_multi_content_research.py`**

`8_` is kept in the repo as a stable reference point and teaching comparison.

---

## Running Locally

### Prerequisites
- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/)
- Optional: [Firecrawl API key](https://www.firecrawl.dev/) for JavaScript-rendered page scraping (500 free pages/month)

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/jspellos/research-agent.git
cd research-agent

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create your .env file
cp .env.example .env
# Then edit .env and add your keys:
#   ANTHROPIC_API_KEY=sk-ant-...
#   FIRECRAWL_API_KEY=fc-...   (optional)

# 4. Run the app
python -m streamlit run 9_multi_content_research.py
```

The app will open at `http://localhost:8501`.

### Project Structure

```
research-agent/
├── 9_multi_content_research.py   # Current main app
├── 8_multi_content_research.py   # Previous stable version
├── profiles/                     # YAML profile configs
│   ├── content_creator.yaml
│   ├── linkedin.yaml
│   └── venue_search.yaml         # (example — add your own)
├── Output/                       # Generated files land here (git-ignored)
├── requirements.txt
└── .env                          # Local secrets (never committed)
```

---

## Deploying to Streamlit Cloud

### First-time deploy

1. Push the repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in
3. Click **Create app**
4. Set **Main file path** to `9_multi_content_research.py`
5. Open **Advanced settings → Secrets** and add:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
FIRECRAWL_API_KEY = "fc-..."
```

6. Click **Deploy**

### Updating the app

Streamlit Cloud watches your GitHub repo and redeploys automatically on every push. The workflow is:

```bash
# Edit locally in VSCode
git add .
git commit -m "describe your change"
git push
# Streamlit Cloud picks up the change within ~30 seconds
```

### Switching to a new main file

Streamlit Cloud does not let you change the main file path after deployment. To point at a new file version:
- Either rename the new file to match the existing filename before pushing, **or**
- Delete the app in the Streamlit dashboard and redeploy pointing at the new file (you'll need to re-enter secrets)

---

## Adding a New Profile

Profiles live in the `profiles/` folder as `.yaml` files. The app loads them automatically — no code changes needed.

Duplicate an existing profile as a starting point, change the `name`, `icon`, `description`, and system prompts for each agent, then push to GitHub.

---

## Requirements

```
anthropic
streamlit
python-dotenv
requests
pyyaml
python-docx
PyPDF2
youtube-transcript-api
firecrawl
```
