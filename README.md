# Yakutlar - Turkic Machine Translation

A Gradio-based web application for translating English text to various Turkic languages using state-of-the-art neural machine translation models (NLLB and MADLAD/T5).

## Supported Languages

The app supports translation to the following Turkic languages:

### NLLB-supported languages:
- Azerbaijani (North & South)
- Bashkir
- Crimean Tatar
- Kazakh
- Kyrgyz
- Tatar
- Turkmen
- Uzbek (Latin)
- Uyghur (Arabic)

### MADLAD/T5-supported languages:
- Chuvash
- Yakut (Sakha)
- Karachay-Balkar
- Gagauz
- Nogai
- Tuvinian
- Karakalpak
- Southern Altai

## Run in GitHub Codespaces

The easiest way to run this app is using GitHub Codespaces - perfect for one-tap usage from mobile devices like iPhone:

### Steps:
1. Go to the repository on GitHub
2. Click **Code** → **Codespaces** → **Create codespace on main**
3. Wait for the Codespace to load and dependencies to install
4. The app will start automatically in the background
5. Go to the **PORTS** panel in VS Code
6. Find the "Yakutlar App" port (7860)
7. Set visibility to **Public** (click the lock icon)
8. Click the URL to open the app on your device

### Optional Authentication:
For secure access, you can set authentication:
```bash
export GRADIO_AUTH="username:password"
# Then restart the app
bash .devcontainer/start.sh
```

## Local Installation

If you prefer to run locally:

```bash
# Clone the repository
git clone https://github.com/engo05/yakutlar.git
cd yakutlar

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

The app will be available at `http://localhost:7860`

## Features

- **Smart Backend Selection**: Automatically chooses the best model for each language
- **Advanced Decoding Options**: Beam search, repetition penalty, length penalty
- **Multiple Translation Modes**: Dictionary lookup for short phrases, pivot translation through Russian
- **Quality Assessment**: Optional COMET evaluation and roundtrip translation scoring
- **Alternative Translations**: View multiple translation hypotheses
- **Comparison Mode**: Side-by-side comparison of different models

## Usage Tips

- For best results with **Chuvash** and **Yakut (Sakha)**, use higher beam sizes (24-32)
- Enable **Dictionary mode** for short phrases (≤2 words) to get better translations
- Use **Russian pivot** for potentially better quality with some languages
- The app includes automatic script cleanup for Cyrillic-writing languages

## Models

This app uses several pre-trained models:
- **NLLB-200**: Facebook's multilingual translation model
- **MADLAD-400**: Google's massively multilingual machine translation model
- **Opus-MT**: Helsinki-NLP's translation models for specific language pairs

Note: Model files need to be downloaded separately and placed in the `models/` directory.