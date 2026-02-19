from anthropic import Anthropic
from dotenv import load_dotenv
from datetime import datetime
load_dotenv()

client = Anthropic()

# ============================================================
# AGENT 1: THE RESEARCHER
# Only job: search the web and produce structured research notes
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


# ============================================================
# AGENT 2: THE WRITER
# Only job: take research notes and produce content in 3 formats
# ============================================================
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
# THE ENGINE: Run any agent with streaming
# ============================================================
def run_agent(system_prompt, user_message, tools=None, label="Agent"):
    """Run a single agent with streaming. Returns the final text output."""

    messages = [{"role": "user", "content": user_message}]

    # Build the API parameters
    params = {
        "model": "claude-sonnet-4-5-20250929",
        "max_tokens": 4096,
        "system": system_prompt,
        "messages": messages,
    }
    if tools:
        params["tools"] = tools

    print(f"\n{'='*50}")
    print(f"  {label} is working...")
    print(f"{'='*50}\n")

    # First streaming call
    with client.messages.stream(**params) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
        response = stream.get_final_message()

    # Agentic loop — keep going if the agent is using tools
    while response.stop_reason != "end_turn":
        messages.append({"role": "assistant", "content": response.content})

        print("\n[Searching...]", flush=True)
        params["messages"] = messages

        with client.messages.stream(**params) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
            response = stream.get_final_message()

    # Extract all text from the final response
    result = ""
    for block in response.content:
        if hasattr(block, "text"):
            result += block.text

    print(f"\n\n{'='*50}")
    print(f"  {label} finished.")
    print(f"{'='*50}\n")

    return result

# ============================================================
# THE ORCHESTRATOR: Connects the agents in sequence
# ============================================================
while True:
    userQ = input("\nEnter a topic (or 'quit'): ")

    if userQ.lower() == "quit":
        break

    # Check if the user has a source file to include
    source_file = input("Source file path (or press Enter to skip): ").strip()

    source_content = ""
    if source_file:
        try:
            with open(source_file, "r", encoding="utf-8") as f:
                source_content = f.read()
            print(f"\nLoaded {len(source_content)} characters from {source_file}")
        except FileNotFoundError:
            print(f"File not found: {source_file} — continuing without source material.")
        except Exception as e:
            print(f"Error reading file: {e} — continuing without source material.")

    # Build the Research Agent's input
    if source_content:
        research_input = f"{userQ}\n\nHere is source material to incorporate:\n\n{source_content}"
    else:
        research_input = userQ

    # Step 1: Research Agent gathers information
    research_notes = run_agent(
        system_prompt=research_prompt,
        user_message=research_input,
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}],
        label="RESEARCH AGENT"
    )

    # Step 2: Writer Agent creates content from the research
    written_content = run_agent(
        system_prompt=writer_prompt,
        user_message=f"Here are the research notes to work from:\n\n{research_notes}",
        tools=None,
        label="WRITER AGENT"
        )

    # Save both outputs to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"output_{timestamp}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write("# RESEARCH NOTES\n\n")
        f.write(research_notes)
        f.write("\n\n---\n\n# WRITTEN CONTENT\n\n")
        f.write(written_content)

    print(f"\nSaved to {filename}")
    print("\n---\n")