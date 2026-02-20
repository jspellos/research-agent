from anthropic import Anthropic
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime

client = Anthropic()

# This is the key — a detailed system prompt that defines the workflow
system_prompt = """You are a research and content creation agent specializing in AI and technology trends.

When given a topic, follow this exact workflow:

STEP 1 - RESEARCH: Analyze the topic thoroughly. Identify 3-5 key themes, 
recent developments, and why this matters to professionals. Present your 
findings as structured research notes.

STEP 2 - PATREON ARTICLE: Write a 400-600 word article suitable for Patreon. 
Tone should be informative but conversational, like you're explaining to a 
smart friend. Include a compelling headline. Use paragraphs, not bullet points.

STEP 3 - LINKEDIN POST: Write a 150-200 word LinkedIn post based on the same 
research. Professional tone, include a hook in the first line, end with a 
question or call to action. Include 3-5 relevant hashtags.

STEP 4 - INSTAGRAM CAPTION: Write 1-2 punchy sentences maximum. 
Casual, bold, attention-grabbing. Add 5-8 relevant hashtags on a 
separate line. This should feel like a scroll-stopping hook, not a summary.

Clearly label each section with === RESEARCH NOTES ===, === PATREON ARTICLE ===, 
=== LINKEDIN POST ===, and === INSTAGRAM CAPTION ===

IMPORTANT: The industry context may change based on the audience. Default to 
technology and AI trends, but adapt your language and examples if a specific 
industry is mentioned (e.g., meeting planning, construction, healthcare)."""

messages = []

while True:
    userQ = input("You: ")

    if userQ.lower() == "quit":
        break

    messages.append({"role": "user", "content": userQ})

    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=4096,
        system=system_prompt,
        messages=messages,
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}]
    )


    while response.stop_reason != "end_turn":
        messages.append({"role": "assistant", "content": response.content})
        response = client.messages.create(
            model="claude-sonnet-4-5-20250929",
            max_tokens=4096,
            system=system_prompt,
            messages=messages,
            tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}]
        )   

    reply = ""
    for block in response.content:
        if block.type == "text":
            reply += block.text

    print("Claude:", reply)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"output_{timestamp}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(reply)
    print(f"[Saved to {filename}]")

    messages.append({"role": "assistant", "content": response.content})