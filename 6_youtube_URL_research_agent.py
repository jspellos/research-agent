# Run it this way: python -m streamlit run 6_youtube_URL_research_agent.py

from anthropic import Anthropic
from dotenv import load_dotenv
from datetime import datetime
import streamlit as st

load_dotenv()

client = Anthropic()

# ============================================================
# AGENT PROMPTS
# ============================================================
research_prompt = """You are a research specialist. Your ONLY job is to research 
a topic thoroughly and produce structured research notes.

You may receive multiple types of source material:
- YouTube video transcripts
- Content from specific web pages
- Uploaded documents (PDFs, text files)
- Preferred sources to prioritize in your web searches

When given a topic:
1. Review ALL provided source material first
2. Use web search to fill gaps and find additional current information
3. When preferred sources are specified, prioritize searching those publications
4. Identify 3-5 key themes and recent developments
5. Note specific facts, statistics, quotes, and sources
6. Explain why this matters to professionals

Output format:
=== RESEARCH NOTES ===
Topic: [topic]
Date: [today's date]

SOURCES REVIEWED:
[List all sources — YouTube videos, web pages, uploaded files, and web searches]

KEY FINDINGS:
[Numbered findings with specific details and sources]

INDUSTRY IMPLICATIONS:
[Why this matters, who it affects, what's changing]

NOTABLE QUOTES/STATS:
[Specific data points worth citing]

Do NOT write articles, posts, or any polished content. 
Your job is ONLY to gather and organize the raw research."""

writer_prompt = """You are a professional content writer. You will receive 
research notes from a research team. Your job is to transform those notes 
into polished content in three formats.

IMPORTANT: Do NOT search the web. Do NOT do your own research. Work ONLY 
from the research notes you are given.

From the research notes, produce:

1. PATREON ARTICLE (400-600 words)
   - Informative but conversational, like explaining to a smart friend
   - Compelling headline
   - Use paragraphs, not bullet points

2. LINKEDIN POST (150-200 words)
   - Professional tone
   - Hook in the first line
   - End with a question or call to action
   - Include 3-5 relevant hashtags

3. INSTAGRAM CAPTION (50-75 words)
   - Casual, punchy, emoji-friendly
   - Include relevant hashtags

Label each section clearly with === PATREON ARTICLE ===, 
=== LINKEDIN POST ===, and === INSTAGRAM CAPTION ===

The industry context may change based on the audience. Default to 
technology and AI trends, but adapt your language and examples if a 
specific industry is mentioned."""


# ============================================================
# HELPER: Fetch YouTube transcripts
# ============================================================
def get_youtube_transcript(url):
    """Extract transcript text from a YouTube video URL."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        # Extract video ID from various YouTube URL formats
        video_id = None
        if "youtu.be/" in url:
            video_id = url.split("youtu.be/")[1].split("?")[0]
        elif "v=" in url:
            video_id = url.split("v=")[1].split("&")[0]
        elif "shorts/" in url:
            video_id = url.split("shorts/")[1].split("?")[0]

        if not video_id:
            return f"[Could not extract video ID from: {url}]"

        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id)
        text = " ".join([snippet.text for snippet in transcript])
        return text

    except Exception as e:
        return f"[Error getting transcript for {url}: {str(e)}]"


# ============================================================
# HELPER: Fetch webpage content
# ============================================================
def get_webpage_content(url):
    """Extract text content from a webpage URL."""
    try:
        import requests
        from html.parser import HTMLParser

        class TextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.text_parts = []
                self.skip_tags = {"script", "style", "nav", "footer", "header"}
                self.current_skip = False

            def handle_starttag(self, tag, attrs):
                if tag in self.skip_tags:
                    self.current_skip = True

            def handle_endtag(self, tag):
                if tag in self.skip_tags:
                    self.current_skip = False

            def handle_data(self, data):
                if not self.current_skip:
                    stripped = data.strip()
                    if stripped:
                        self.text_parts.append(stripped)

        response = requests.get(url, timeout=15, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        response.raise_for_status()

        parser = TextExtractor()
        parser.feed(response.text)
        return " ".join(parser.text_parts)

    except Exception as e:
        return f"[Error reading {url}: {str(e)}]"


# ============================================================
# THE ENGINE: Run any agent with streaming for Streamlit
# ============================================================
def run_agent_stream(system_prompt, user_message, tools=None):
    """Run a single agent. Yields text chunks for streaming."""

    messages = [{"role": "user", "content": user_message}]

    params = {
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 4096,
        "system": system_prompt,
        "messages": messages,
    }
    if tools:
        params["tools"] = tools

    with client.messages.stream(**params) as stream:
        for text in stream.text_stream:
            yield text
        response = stream.get_final_message()

    while response.stop_reason != "end_turn":
        messages.append({"role": "assistant", "content": response.content})
        params["messages"] = messages

        with client.messages.stream(**params) as stream:
            for text in stream.text_stream:
                yield text
            response = stream.get_final_message()


# ============================================================
# STREAMLIT INTERFACE
# ============================================================
st.set_page_config(page_title="Research & Content Agent", layout="wide")
st.title("🔬 Research & Content Agent")
st.markdown("Enter a topic and provide sources. The Research Agent will gather information, then the Writer Agent will create your content.")

# --- INPUT SECTION ---
topic = st.text_input("Enter a topic:", placeholder="e.g., Latest AI agent developments in 2026")

# Two columns for source inputs
col1, col2 = st.columns(2)

with col1:
    youtube_urls = st.text_area(
        "YouTube URLs (one per line)",
        placeholder="https://youtube.com/watch?v=...\nhttps://youtu.be/...",
        height=120
    )

    uploaded_file = st.file_uploader(
        "Upload source material (optional)",
        type=["txt", "md", "csv", "pdf"]
    )

with col2:
    website_urls = st.text_area(
        "Website URLs to read (one per line)",
        placeholder="https://example.com/article\nhttps://news.site.com/story",
        height=120
    )

    preferred_sources = st.text_area(
        "Preferred sources for web search (optional)",
        placeholder="e.g., Skift, MeetingsNet, PCMA, TechCrunch, Ars Technica",
        height=68
    )

# --- RUN BUTTON ---
if st.button("Run Agents", type="primary") and topic:

    # ============================
    # GATHER ALL SOURCE MATERIAL
    # ============================
    source_sections = []

    # 1. YouTube transcripts
    if youtube_urls.strip():
        urls = [u.strip() for u in youtube_urls.strip().split("\n") if u.strip()]
        st.info(f"Fetching {len(urls)} YouTube transcript(s)...")

        for url in urls:
            with st.spinner(f"Reading: {url}"):
                transcript = get_youtube_transcript(url)
                source_sections.append(f"=== YOUTUBE TRANSCRIPT: {url} ===\n{transcript}")

        st.success(f"Loaded {len(urls)} YouTube transcript(s)")

    # 2. Website content
    if website_urls.strip():
        urls = [u.strip() for u in website_urls.strip().split("\n") if u.strip()]
        st.info(f"Reading {len(urls)} webpage(s)...")

        for url in urls:
            with st.spinner(f"Reading: {url}"):
                content = get_webpage_content(url)
                source_sections.append(f"=== WEBPAGE CONTENT: {url} ===\n{content}")

        st.success(f"Loaded {len(urls)} webpage(s)")

    # 3. Uploaded file
    if uploaded_file:
        if uploaded_file.name.endswith(".pdf"):
            import PyPDF2
            import io
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            file_content = ""
            for page in pdf_reader.pages:
                file_content += page.extract_text() + "\n"
        else:
            file_content = uploaded_file.read().decode("utf-8")

        source_sections.append(f"=== UPLOADED FILE: {uploaded_file.name} ===\n{file_content}")
        st.success(f"Loaded {len(file_content)} characters from {uploaded_file.name}")

    # 4. Preferred sources guidance
    preferred_guidance = ""
    if preferred_sources.strip():
        preferred_guidance = f"\n\nPREFERRED SOURCES: When searching the web, prioritize these sources: {preferred_sources.strip()}"

    # ============================
    # BUILD RESEARCH INPUT
    # ============================
    research_input = f"Topic: {topic}{preferred_guidance}"

    if source_sections:
        research_input += "\n\nSOURCE MATERIAL PROVIDED:\n\n" + "\n\n".join(source_sections)

    # ============================
    # RUN RESEARCH AGENT
    # ============================
    st.header("📋 Research Agent")
    st.caption("Reviewing sources and searching the web...")

    research_container = st.empty()
    research_text = ""
    for chunk in run_agent_stream(
        system_prompt=research_prompt,
        user_message=research_input,
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}],
    ):
        research_text += chunk
        research_container.markdown(research_text)

    st.success("Research complete!")

    # ============================
    # RUN WRITER AGENT
    # ============================
    st.header("✍️ Writer Agent")
    st.caption("Creating content from research notes...")

    writer_container = st.empty()
    writer_text = ""
    for chunk in run_agent_stream(
        system_prompt=writer_prompt,
        user_message=f"Here are the research notes to work from:\n\n{research_text}",
        tools=None,
    ):
        writer_text += chunk
        writer_container.markdown(writer_text)

    st.success("Content complete!")

    # ============================
    # SAVE AND DOWNLOAD
    # ============================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"output_{timestamp}.md"
    full_output = f"# RESEARCH NOTES\n\n{research_text}\n\n---\n\n# WRITTEN CONTENT\n\n{writer_text}"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(full_output)

    st.download_button(
        label="📥 Download Output",
        data=full_output,
        file_name=filename,
        mime="text/markdown"
    )

    st.info(f"Also saved locally to {filename}")