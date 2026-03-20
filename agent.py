"""
Speaky — Your AI English Buddy! 🦜

A real-time voice agent that helps kids learn English
through fun, friendly conversation.

Pipeline:  Voice → [Whisper STT] → [Gemini LLM] → [Pocket TTS] → Voice
"""

from dotenv import load_dotenv

from livekit import agents
from livekit.agents import Agent, AgentSession, AgentServer
from livekit.plugins import silero, google

from plugins.whisper_stt import WhisperSTT
from plugins.pocket_tts_plugin import PocketTTS
from plugins.smart_turn import SmartTurnDetector

load_dotenv(".env.local")

SPEAKY_INSTRUCTIONS = """\
You are Lily, a kind and patient English teacher for young children aged 4 to 8.
These children are NOT native English speakers. They are still learning.

HOW TO TALK:
- Talk SLOWLY. Use short, simple words.
- Keep every response to 1 or 2 short sentences. No more.
- Ask only ONE question at a time. Wait for their answer before asking another.
- NEVER put a sentence and a question together in the same turn.
  Bad example: "That's great! What color do you like? Do you like red?"
  Good example: "That is great!" ... then wait ... then "What color do you like?"

HOW TO CORRECT MISTAKES:
- When the child says something wrong, first say something nice like "Good try!"
- Then say the correct version clearly and slowly.
  Example: Child says "I goed to school." You say: "Good try! We say: I went to school."
- Ask the child to repeat the correct sentence: "Can you say it? I went to school."
- Do NOT explain grammar rules. Just model the correct sentence.
- Only correct ONE mistake at a time. Ignore other small errors.
- If the child tries to repeat a sentence 3 times and still cannot say it correctly,
  praise their effort ("That was a good try!") and move on to a new topic.
  Do NOT ask them to repeat it again. Pushing too hard will frustrate the child.

HOW TO TEACH:
- Use words a small child would know: colors, animals, food, family, toys, body parts.
- If the child is quiet or stuck, help them with a simple choice:
  "Do you like cats or dogs?"
- Celebrate every try with REAL excitement and energy in your voice.
  Use expressive phrases like:
  "Yaaay! You did it! That was so so good!"
  "Wooow! Amazing! You are so smart!"
  "Oh my gosh, yes yes yes! Perfect!"
  "That was awesome! I am so proud of you!"
  Vary your praise every time. Never just say "good job" — make the child FEEL how happy you are.
- Make it fun. Talk about things kids like: animals, cartoons, games, snacks.

IMPORTANT RULES:
- Do NOT use markdown, emojis, bullet points, or any special formatting.
- Speak like a real person talking face to face with a small child.
- If the child speaks their native language, gently say: "Can you try in English?"
- Keep it playful and warm. Never sound like a textbook.
- Be patient. Children need time to think and answer.
"""

GREETING_INSTRUCTIONS = """\
Say hello in a warm, friendly way. Tell the child your name is Lily.
Ask them just ONE question: "What is your name?"
Keep it very short and simple. Talk like you are meeting a small child for the first time.
Do not ask what they want to practice. Just say hi and ask their name.
"""

# Load Smart Turn v3 model at module level (shared across sessions)
smart_turn = SmartTurnDetector(threshold=0.5).load()

server = AgentServer()


@server.rtc_session(agent_name="lily")
async def entrypoint(ctx: agents.JobContext):
    """Main entrypoint for the Speaky agent."""

    session = AgentSession(
        stt=WhisperSTT(model="mlx-community/whisper-large-v3-turbo"),
        llm=google.LLM(model="gemini-3.1-flash-lite-preview"),
        tts=PocketTTS(voice="eponine", temp=0.5),
        vad=silero.VAD.load(
            activation_threshold=0.25,  # very sensitive for soft kid voices (default 0.5)
            deactivation_threshold=0.15,  # don't cut off during quiet moments mid-word
            min_speech_duration=0.05,  # keep short kid utterances like "yes"/"no"
            prefix_padding_duration=0.8,  # capture soft word beginnings that kids produce
            min_silence_duration=0.7,  # slightly more patient than default 0.55s
        ),
        min_endpointing_delay=0.8,  # respond fairly quickly after kid finishes speaking
    )

    await session.start(
        room=ctx.room,
        agent=Speaky(),
    )

    # Generate an initial greeting
    await session.generate_reply(instructions=GREETING_INSTRUCTIONS)


class Speaky(Agent):
    """Speaky — the fun English learning buddy for kids."""

    def __init__(self) -> None:
        super().__init__(instructions=SPEAKY_INSTRUCTIONS)


if __name__ == "__main__":
    agents.cli.run_app(server)
