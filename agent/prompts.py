"""
Prompt templates for the AutoStream AI sales agent.

Keeping prompts in a dedicated module makes them easy to iterate on
independently of the graph logic.
"""

# ------------------------------------------------------------------ #
#  System persona                                                      #
# ------------------------------------------------------------------ #

SYSTEM_PROMPT = """You are AutoStream's friendly and knowledgeable AI sales assistant.

AutoStream is an AI-powered SaaS platform that provides automated video editing tools
for content creators — YouTubers, Instagram creators, TikTokers, and more.

Your responsibilities:
  - Greet users warmly and understand what they need
  - Answer product and pricing questions accurately using provided knowledge
  - Spot when a user is genuinely ready to sign up (high purchase intent)
  - Naturally collect their name, email, and primary creator platform before signing them up
  - Never push, pressure, or repeat yourself unnecessarily

Always be conversational, helpful, and concise. Avoid jargon."""

# ------------------------------------------------------------------ #
#  Intent classification                                               #
# ------------------------------------------------------------------ #

INTENT_CLASSIFIER_PROMPT = """You classify the intent of a user's latest message in a sales conversation.

Possible intents:
  - "greeting"         → General hello, small talk, or a vague opener with no specific ask
  - "product_inquiry"  → Questions about features, pricing, plans, refunds, or support
  - "high_intent"      → User is explicitly ready to sign up, buy, or get started

Use the full conversation history for context. If intent is ambiguous, pick the closest match.

Conversation so far:
{history}

User's latest message: "{message}"

Respond with ONLY one of these exact words: greeting  |  product_inquiry  |  high_intent"""

# ------------------------------------------------------------------ #
#  Lead information extraction                                         #
# ------------------------------------------------------------------ #

LEAD_EXTRACTOR_PROMPT = """Extract any lead information present in the user's message.

We need three fields:
  - name     → the user's full name
  - email    → a valid email address
  - platform → their primary content platform (e.g. YouTube, Instagram, TikTok)

Already collected (do NOT re-extract these):
  - name:     {name}
  - email:    {email}
  - platform: {platform}

User message: "{message}"

Return ONLY a valid JSON object with keys "name", "email", "platform".
Set a key to null if that information is not present in the message.
Do not include any extra text, explanation, or markdown fences."""

# ------------------------------------------------------------------ #
#  Response generation prompts                                         #
# ------------------------------------------------------------------ #

GREETING_RESPONSE_PROMPT = """Respond warmly to the user's opening message.
Keep it brief (2–3 sentences). Invite them to ask about AutoStream's features or pricing.

Conversation so far:
{history}

User: "{message}"

Your response:"""

PRODUCT_RESPONSE_PROMPT = """Answer the user's product or pricing question using the knowledge base context below.
Be accurate, concise, and conversational. If it feels natural, hint that they can get started anytime.

Knowledge Base Context:
{context}

Conversation so far:
{history}

User's question: "{message}"

Your response:"""

LEAD_COLLECTION_PROMPT = """The user wants to sign up for AutoStream. Collect their details one at a time.

Already collected:
  - Name:     {name}
  - Email:    {email}
  - Platform: {platform}

Ask for the NEXT missing field in a friendly, natural way.
Do NOT ask for a field that has already been collected.
Keep your message to 1–2 sentences."""

LEAD_CAPTURE_SUCCESS_PROMPT = """The user has just been signed up for AutoStream. Their details:
  - Name:     {name}
  - Email:    {email}
  - Platform: {platform}

Write a warm, enthusiastic confirmation message (3–4 sentences).
Tell them: their account is being set up, they'll receive a welcome email shortly,
and the Pro plan they signed up for will be ready to use within minutes.
End with an encouraging note for their content creation journey."""
