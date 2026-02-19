from anthropic import Anthropic
from dotenv import load_dotenv
from datetime import datetime
load_dotenv()

client = Anthropic()

system_prompt = """You are a research and content creation agent specializing in providing in-depth company information.

When given a topic, follow this exact workflow:

STEP 1 - RESEARCH: Analyze the topic thoroughly. Make sure the research ties into the potential
and actual use of Generative AI for the company. Identify 3-5 key themes, 
recent news, and why this information is important to clients and the industry. Present your 
findings as structured research notes.

STEP 2 - REPORT WRITER - Provide a 5-10 page report on the history of the company/topic provided.  
Focus on company stengths and opportunities, especially regarding the use of Generative AI technologies.

Clearly label each section with === RESEARCH NOTES ===, === COMPANY OVERVIEW ===

IMPORTANT: The industry context may change based on the audience. Default to 
technology and AI trends, but adapt your language and examples if a specific 
industry is mentioned (e.g., meeting planning, construction, healthcare)."""

messages = []

while True:
    userQ = input("\nYou: ")

    if userQ.lower() == "quit":
        break

    messages.append({"role": "user", "content": userQ})

    # --- STREAMING CHANGE #1 ---
    # Instead of client.messages.create(), we use client.messages.stream()
    # wrapped in a "with" block. This streams text in real time.
    print("\nClaude: ", end="", flush=True)

    with client.messages.stream(
        model="claude-sonnet-4-5-20250929",
        max_tokens=4096,
        system=system_prompt,
        messages=messages,
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}]
    ) as stream:
        # --- STREAMING CHANGE #2 ---
        # Print each chunk of text as it arrives (the "typewriter" effect)
        for text in stream.text_stream:
            print(text, end="", flush=True)

        # --- STREAMING CHANGE #3 ---
        # After streaming finishes, get the full response object
        # so we can check stop_reason and handle tool use
        response = stream.get_final_message()

    # The agentic loop - same logic as before, but now with streaming inside
    while response.stop_reason != "end_turn":
        messages.append({"role": "assistant", "content": response.content})

        print("\n[Searching...]", flush=True)

        with client.messages.stream(
            model="claude-sonnet-4-5-20250929",
            max_tokens=4096,
            system=system_prompt,
            messages=messages,
            tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}]
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
            response = stream.get_final_message()

    print("\n\n---\n")

    # Save full response to conversation history (for follow-up questions)
    messages.append({"role": "assistant", "content": response.content})

    # Extract all text blocks for file saving
    reply = ""
    for block in response.content:
        if hasattr(block, "text"):
            reply += block.text

    # Auto-save to timestamped file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"output_{timestamp}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(reply)
    print(f"Saved to {filename}")