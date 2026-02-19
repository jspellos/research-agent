from anthropic import Anthropic
from dotenv import load_dotenv
load_dotenv()

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

STEP 4 - INSTAGRAM CAPTION: Write a 50-75 word Instagram caption. Casual, 
punchy, emoji-friendly. Include relevant hashtags.

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
        messages=messages
    )

    reply = response.content[0].text
    print("Claude:", reply)
    print("\n---\n")

    messages.append({"role": "assistant", "content": reply})