# Use this to run the program:  python -m streamlit run 5_streamlit_agent.py

from anthropic import Anthropic
from dotenv import load_dotenv
from datetime import datetime
import streamlit as st

load_dotenv()

client = Anthropic()

# ============================================================
# AGENT PROMPTS (same as 4_multi_agent.py)
# ============================================================
research_prompt = """You are a research specialist. Your ONLY job is to research 
a topic thoroughly using web search and produce structured research notes.

When given a topic:
1. Search for the most current information available
2. Identify 3-5 key themes and recent developments
3. Note specific facts, statistics, quotes, and sources
4. Explain why this matters to professionals

Output format:
=== RESEARCH NOTES ===
Topic: [topic]
Date: [today's date]

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
# THE ENGINE: Run any agent with streaming for Streamlit
# ============================================================
def run_agent_stream(system_prompt, user_message, tools=None):
    """Run a single agent. Yields text chunks for streaming, returns full text."""

    messages = [{"role": "user", "content": user_message}]

    params = {
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 4096,
        "system": system_prompt,
        "messages": messages,
    }
    if tools:
        params["tools"] = tools

    # First streaming call
    with client.messages.stream(**params) as stream:
        for text in stream.text_stream:
            yield text
        response = stream.get_final_message()

    # Agentic loop — keep going if the agent is using tools
    while response.stop_reason != "end_turn":
        messages.append({"role": "assistant", "content": response.content})
        params["messages"] = messages

        with client.messages.stream(**params) as stream:
            for text in stream.text_stream:
                yield text
            response = stream.get_final_message()


def run_agent_full(system_prompt, user_message, tools=None):
    """Run a single agent. Returns the complete text output (no streaming)."""

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
        response = stream.get_final_message()

    while response.stop_reason != "end_turn":
        messages.append({"role": "assistant", "content": response.content})
        params["messages"] = messages

        with client.messages.stream(**params) as stream:
            response = stream.get_final_message()

    result = ""
    for block in response.content:
        if hasattr(block, "text"):
            result += block.text

    return result


# ============================================================
# STREAMLIT INTERFACE
# ============================================================
st.set_page_config(page_title="Research & Content Agent", layout="wide")
st.title("🔬 MPI Upstate New York - Research & Content Agent")
st.markdown("Enter a topic below. The Research Agent will search the web, then the Writer Agent will create your content.")

# Topic input
topic = st.text_input("Enter a topic:", placeholder="e.g., AI trends in meeting planning for 2026")

# Optional file upload
uploaded_file = st.file_uploader("Upload source material (optional)", type=["txt", "md", "csv", "pdf"])

# Run button
if st.button("Run Agents", type="primary") and topic:

    # Build research input
    research_input = topic
    if uploaded_file:
        if uploaded_file.name.endswith(".pdf"):
            import PyPDF2
            import io
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            source_content = ""
            for page in pdf_reader.pages:
                source_content += page.extract_text() + "\n"
        else:
            source_content = uploaded_file.read().decode("utf-8")
        research_input = f"{topic}\n\nHere is source material to incorporate:\n\n{source_content}"
        st.success(f"Loaded {len(source_content)} characters from {uploaded_file.name}")

    # --- RESEARCH AGENT ---
    st.header("📋 Research Agent")
    st.caption("Searching the web and compiling research notes...")

    # Stream the research output
    research_container = st.empty()
    research_text = ""
    for chunk in run_agent_stream(
        system_prompt=research_prompt,
        user_message=research_input,
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}]
    ):
        research_text += chunk
        research_container.markdown(research_text)

    st.success("Research complete!")

    # --- WRITER AGENT ---
    st.header("✍️ Writer Agent")
    st.caption("Creating content from research notes...")

    # Stream the writer output
    writer_container = st.empty()
    writer_text = ""
    for chunk in run_agent_stream(
        system_prompt=writer_prompt,
        user_message=f"Here are the research notes to work from:\n\n{research_text}",
        tools=None
    ):
        writer_text += chunk
        writer_container.markdown(writer_text)

    st.success("Content complete!")

    # --- SAVE TO FILE ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"output_{timestamp}.md"
    full_output = f"# RESEARCH NOTES\n\n{research_text}\n\n---\n\n# WRITTEN CONTENT\n\n{writer_text}"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(full_output)

    # Download button
    st.download_button(
        label="📥 Download Output",
        data=full_output,
        file_name=filename,
        mime="text/markdown"
    )

    st.info(f"Also saved locally to {filename}")
