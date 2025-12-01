import streamlit as st
import PyPDF2
import os
import json
import requests
import base64
import tempfile
from gtts import gTTS

# Optional voice input
try:
    import speech_recognition as sr
    HAS_SR = True
except:
    HAS_SR = False

# Optional RAG embeddings
try:
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer
    HAS_RAG = True
except:
    HAS_RAG = False

# Optional web scraping
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except:
    HAS_BS4 = False

# -----------------------------
# PAGE CONFIG
st.set_page_config(page_title="AI Assistant Pro", page_icon="🤖", layout="wide")
st.title("🤖 AI Assistant Pro")
st.caption("Chat | Voice | Files | Memory | Web + RAG + Images | Multi-language")

# -----------------------------
# SERVER-SIDE API KEY
api_key = os.getenv("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")  # <-- Key is stored in environment variable
if not api_key:
    st.error("OpenRouter API key is missing! Set OPENROUTER_API_KEY environment variable.")
model_name = st.selectbox("Select Model", ["meta-llama/llama-3.1-8b-instruct",
                                           "qwen-7b", "mistral-7b-instruct"])
dark_mode = st.checkbox("Dark Mode", False)
enable_voice = st.checkbox("Enable Voice Output", True)

# -----------------------------
# SESSION STATE
HISTORY_FILE = "chat_history.json"

if "chat_history" not in st.session_state:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            st.session_state.chat_history = json.load(f)
    else:
        st.session_state.chat_history = []

if "file_memory" not in st.session_state:
    st.session_state.file_memory = ""

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
    st.session_state.vector_texts = []

if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None

if "show_plus_menu" not in st.session_state:
    st.session_state.show_plus_menu = False

if "active_chat" not in st.session_state:
    st.session_state.active_chat = []

if "form_counter" not in st.session_state:
    st.session_state.form_counter = 0

# -----------------------------
# HELPER FUNCTIONS
def save_history():
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(st.session_state.chat_history, f, ensure_ascii=False, indent=2)
    except:
        pass

def ask_openrouter(messages, model=model_name):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"model": model, "messages": messages}
    try:
        res = requests.post("https://openrouter.ai/api/v1/chat/completions",
                            headers=headers, json=body, timeout=30)
        res.raise_for_status()
        data = res.json()
        return data["choices"][0]["message"]["content"]
    except:
        return offline_assistant(messages)

def offline_assistant(messages):
    user_msg = messages[-1].get("content", "") if messages else ""
    response = ""
    if st.session_state.file_memory and HAS_RAG and st.session_state.vector_store:
        relevant = query_vector_store(user_msg)
        if relevant:
            response += f"Based on your uploaded file:\n{relevant}\n\n"
    if st.session_state.chat_history:
        response += "Based on previous conversation memory.\n"
    if not response:
        response = "⚠️ Cannot reach OpenRouter. Provide more context or upload a file."
    return response

def extract_text_from_file(uploaded):
    if uploaded.type == "application/pdf":
        reader = PyPDF2.PdfReader(uploaded)
        return "\n".join([page.extract_text() for page in reader.pages])
    else:
        return uploaded.read().decode("utf-8")

def voice_to_text():
    if not HAS_SR:
        st.error("Voice input requires SpeechRecognition and a microphone.")
        return None
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Speak now...")
        audio = r.listen(source)
    try:
        return r.recognize_google(audio)
    except:
        st.error("Voice not recognized")
        return None

def text_to_voice(text):
    try:
        tts = gTTS(text)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmp.name)
        st.audio(tmp.name)
    except:
        st.error("TTS failed")

def add_to_vector_store(text):
    if not HAS_RAG:
        return
    if st.session_state.embedding_model is None:
        st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    sentences = [s for s in text.split("\n") if s.strip()]
    embeddings = st.session_state.embedding_model.encode(sentences)
    embeddings = np.array(embeddings).astype('float32')
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    st.session_state.vector_store = index
    st.session_state.vector_texts = sentences

def query_vector_store(query, top_k=3):
    if not HAS_RAG or st.session_state.vector_store is None:
        return ""
    query_emb = st.session_state.embedding_model.encode([query]).astype('float32')
    D, I = st.session_state.vector_store.search(query_emb, top_k)
    return "\n".join([st.session_state.vector_texts[i] for i in I[0] if i < len(st.session_state.vector_texts)])

def summarize_website(url):
    if not HAS_BS4:
        st.error("BeautifulSoup4 required for web summarization.")
        return ""
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')
        paragraphs = soup.find_all('p')
        return "\n".join([p.get_text() for p in paragraphs])[:3000]
    except:
        return ""

# -----------------------------
# PLUS MENU BUTTON
if st.button("➕"):
    st.session_state.show_plus_menu = not st.session_state.show_plus_menu

# PLUS MENU
if st.session_state.show_plus_menu:
    choice = st.radio("Choose an action:", ["Upload File", "Voice Input", "Image Generation"], horizontal=True)
    if choice == "Upload File":
        uploaded_file = st.file_uploader("Select file (PDF/TXT)", type=["pdf", "txt"])
        if uploaded_file:
            text = extract_text_from_file(uploaded_file)
            st.session_state.file_memory = text
            add_to_vector_store(text)
            st.success("File uploaded and indexed!")
    elif choice == "Voice Input":
        if st.button("Start Voice Input"):
            voice_text = voice_to_text()
            if voice_text:
                st.session_state.chat_history.append({"role":"user","content":voice_text})
                st.success("Voice input added!")
    elif choice == "Image Generation":
        img_prompt = st.text_input("Enter image description:")
        if st.button("Generate Image"):
            st.info("Image generation placeholder - connect to OpenRouter Image API")

# -----------------------------
# CHAT INPUT FORM
form_key = f"chat_form_{st.session_state.form_counter}"
st.session_state.form_counter += 1

with st.form(key=form_key, clear_on_submit=True):
    user_message = st.text_input("Type your message...", key=f"msg_input_{form_key}", placeholder="Write your question here...")
    submitted = st.form_submit_button("Send")

if submitted and user_message.strip():
    st.session_state.chat_history.append({"role":"user","content":user_message})
    messages = [{"role":"system","content":"You are a helpful AI assistant."}]
    if st.session_state.active_chat:
        messages.extend(st.session_state.active_chat)
    if st.session_state.file_memory:
        relevant = query_vector_store(user_message)
        context_text = f"User uploaded file content (relevant):\n{relevant}" if relevant else f"User uploaded file content:\n{st.session_state.file_memory}"
        messages.append({"role":"system","content":context_text})
    messages.append({"role":"user","content":user_message})
    reply = ask_openrouter(messages, model_name)
    st.session_state.chat_history.append({"role":"assistant","content":reply})
    st.session_state.active_chat.append({"role":"user","content":user_message})
    st.session_state.active_chat.append({"role":"assistant","content":reply})
    st.write("🤖", reply)
    if enable_voice:
        text_to_voice(reply)
    save_history()
