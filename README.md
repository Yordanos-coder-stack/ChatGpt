# ChatGPT Clone (OpenRouter)

A tiny Streamlit-based ChatGPT-like frontend that sends prompts to OpenRouter (Llama 3.1 / Qwen / Mistral models).

## Setup

1. Create a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

### Windows microphone / pyaudio notes

On Windows, `speech_recognition` relies on a microphone backend like `pyaudio`. Installing `pyaudio` with pip can be difficult due to missing build tools. Use `pipwin` to install a prebuilt wheel:

```powershell
pip install pipwin
pipwin install pyaudio
```

If you don't need voice features, you can skip installing `SpeechRecognition` and `pyaudio`; the app will still run but voice features will be disabled.

### Website summarization

This app uses `beautifulsoup4` to extract text from web pages. If you want the "Summarize Web Page" feature, install:

```powershell
pip install beautifulsoup4
```

## Run

```powershell
streamlit run .\app.py
```

Open the URL printed by Streamlit in your browser (usually `http://localhost:8501`).

## Usage

- Enter your OpenRouter API key in the top input (it is stored in Streamlit's session state).
- Type a message and click `Send` to get a response from the selected model.

## Notes

- This project relies on the OpenRouter API. Make sure your API key has the required permissions.
- If the API response shape differs from expectations, paste a redacted sample here and I can make the parser handle it.
