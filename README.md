
                            ██╗
                           ██╔╝
                    ●  ●  ██╔╝
                     ╲__╱██╔╝
                      ████╔╝
                   ╔══╝ ╚═╗
                  ██  ███  ██
                 ██  █████  ██
                  ██  ███  ██
                   ╚══════╝
              ┌──────────────────┐
              │     S P E A K Y  │
              │  🦜 Your AI      │
              │  English Buddy!  │
              └──────────────────┘


# Speaky 🦜

> **Hey there, superstar!** Speaky is your friendly AI buddy that helps you learn English through fun voice conversations. Just talk — and Speaky talks back! 🎤✨

---

## ✨ What Does Speaky Do?

Speaky is like having a super-patient, always-happy English teacher right in your computer! Here's how it works:

```
🎤 You talk ──→ 🧠 Speaky listens ──→ 💬 Speaky replies ──→ 🔊 You hear it!
```

- 🗣️ **Talk freely** — Speaky understands what you say
- 📝 **Learn words** — Practice vocabulary, grammar, and pronunciation
- 🎉 **Have fun** — Speaky makes learning feel like playing!
- 💪 **Get better** — Speaky gently helps you improve

---

## 🧩 How It Works (The Cool Techy Stuff)

| What | How | Magic ✨ |
|------|-----|---------|
| **Ears** 👂 | Whisper Large v3 Turbo | Listens to your voice super fast |
| **Brain** 🧠 | Google Gemini 2.5 Flash | Thinks of the best reply |
| **Voice** 🔊 | Pocket TTS | Talks back in a friendly voice |
| **Attention** 👀 | Silero VAD | Knows when you're talking |
| **Patience** 🤔 | Smart Turn v3 | Waits until you're done speaking |

> 💡 **Fun fact:** Speaky runs on Apple Silicon, which means it's *super fast* on Mac!

---

## 🚀 Getting Started

### 1. Install everything

```bash
uv sync
```

### 2. Add your secret keys

Create a `.env.local` file:

```env
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your-api-key
LIVEKIT_API_SECRET=your-api-secret
GOOGLE_API_KEY=your-google-api-key
```

🔑 **Where to get keys:**
- **LiveKit** → [cloud.livekit.io](https://cloud.livekit.io) (free!)
- **Google AI** → [ai.google.dev](https://ai.google.dev/)

### 3. Launch Speaky!

```bash
uv run python agent.py dev
```

### 4. Start Talking!

Open the [LiveKit Playground](https://agents-playground.livekit.io/) and say hi to Speaky! 👋

---

## 📁 Project Structure

```
speaky/
├── agent.py              # 🧠 Speaky's brain — the main agent
├── plugins/
│   ├── whisper_stt.py    # 👂 Ears — speech-to-text (MLX Whisper)
│   ├── pocket_tts_plugin.py  # 🔊 Voice — text-to-speech (Pocket TTS)
│   └── smart_turn.py     # 🤔 Patience — knows when you stop talking
├── pyproject.toml        # 📦 Dependencies
├── .env.local            # 🔑 Secret keys (not shared!)
└── README.md             # 📖 You are here!
```

---

## 📝 Good to Know

- **First time?** Models download automatically (~2GB total). Grab a snack! 🍿
- **Mac only?** The STT uses MLX, optimized for Apple Silicon (M1/M2/M3/M4)
- **A bit slow?** STT & TTS run in batch mode — streaming support coming soon!

---

## 🤝 Contributing

Got ideas to make Speaky even cooler? PRs welcome! Let's make English learning awesome for every kid! 🌍

---

<p align="center">
  Made with 💚 for kids who want to speak English like a boss 😎
</p>
]]>
