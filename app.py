import os
import json
import time
from pathlib import Path
from functools import lru_cache
import gradio as gr
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts.prompt import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from chromadb import Client, Settings
from langchain.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
import requests

# Constants
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
PASSAGES_FILE = DATA_DIR / "all_text_passages.json"
EMBEDDINGS_FILE = DATA_DIR / "all_text_embeddings.json"

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY in your .env file")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Cache for API responses
@lru_cache(maxsize=None)
def fetch_cached_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None

# ----------------------
# 1️⃣ SWAPI Data Loading with Caching
# ----------------------

def get_all_data(api_url):
    results = []
    while api_url:
        response = requests.get(api_url)
        if response.status_code == 200:
            data = response.json()
            results.extend(data["results"])
            api_url = data["next"]
        else:
            print(f"Failed to fetch data from {api_url}")
            break
    return results


def process_data_to_text(data, category):
    passages = []
    for item in data:
        if category == "people":
            passage = f"Name: {item.get('name')}\nHeight: {item.get('height')}\nMass: {item.get('mass')}\n" \
                      f"Hair color: {item.get('hair_color')}\nSkin color: {item.get('skin_color')}\n" \
                      f"Eye color: {item.get('eye_color')}\nBirth year: {item.get('birth_year')}\nGender: {item.get('gender')}\n"
        elif category == "films":
            passage = f"Film: {item.get('title')}\nEpisode: {item.get('episode_id')}\n" \
                      f"Director: {item.get('director')}\nProducer: {item.get('producer')}\nRelease date: {item.get('release_date')}\n"
        else:
            passage = f"Category: {category}\n"
            for key, value in item.items():
                if isinstance(value, list):
                    value = ", ".join([str(v) for v in value])
                passage += f"{key.replace('_',' ').capitalize()}: {value}\n"
        passages.append(passage)
    return passages

base_url = "https://swapi.dev/api/"
categories = ["people", "planets", "starships", "vehicles", "species", "films"]
all_text_passages = []

for cat in categories:
    data = get_all_data(base_url + cat + "/")
    passages = process_data_to_text(data, cat)
    all_text_passages.extend(passages)

# Load or create cached data
def load_or_create_data():
    if PASSAGES_FILE.exists() and EMBEDDINGS_FILE.exists():
        try:
            with open(PASSAGES_FILE, "r", encoding="utf-8") as f:
                all_text_passages = json.load(f)
            with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
                embeddings = json.load(f)
            print("✅ Loaded cached data files")
            return all_text_passages, embeddings
        except Exception as e:
            print(f"⚠️ Error loading cached files: {e}")
    
    print("🔄 Fetching and processing SWAPI data...")
    all_text_passages = []
    
    # Fetch data with caching
    base_url = "https://swapi.dev/api/"
    categories = ["people", "planets", "starships", "vehicles", "species", "films"]
    
    for category in categories:
        data = []
        url = base_url + category + "/"
        
        while url:
            cached_response = fetch_cached_data(url)
            if cached_response:
                data.extend(cached_response["results"])
                url = cached_response.get("next")
            else:
                break
                
        passages = process_data_to_text(data, category)
        all_text_passages.extend(passages)
        print(f"✓ Processed {category}: {len(passages)} items")
    
    # Initialize model and compute embeddings
    print("🔄 Computing embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = [model.encode(text).tolist() for text in all_text_passages]
    
    # Save processed data
    DATA_DIR.mkdir(exist_ok=True)
    with open(PASSAGES_FILE, "w", encoding="utf-8") as f:
        json.dump(all_text_passages, f, ensure_ascii=False, indent=2)
    with open(EMBEDDINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(embeddings, f)
    print("✅ Saved processed data to cache")
    
    return all_text_passages, embeddings

# Load data and compute embeddings
all_text_passages, embeddings = load_or_create_data()

# ----------------------
# 2️⃣ Chroma Setup with Persistent Storage
# ----------------------
CHROMA_DIR = Path("chroma_db")
CHROMA_DIR.mkdir(exist_ok=True)

# Initialize Chroma with persistent storage
client = Client(Settings(
    persist_directory=str(CHROMA_DIR),
    is_persistent=True
))

# Get or create collection with batch processing
collection = client.get_or_create_collection(
    name="starwars",
    metadata={"hnsw:space": "cosine"}  # Optimized similarity search
)

# Initialize or update collection with batch processing
collection_size = len(collection.get()["ids"])
if collection_size == 0:
    print("🔄 Initializing Chroma collection...")
    # Process in batches to avoid memory issues
    BATCH_SIZE = 100
    for i in range(0, len(all_text_passages), BATCH_SIZE):
        batch_texts = all_text_passages[i:i + BATCH_SIZE]
        batch_embeddings = embeddings[i:i + BATCH_SIZE]
        batch_ids = [str(j) for j in range(i, i + len(batch_texts))]
        batch_metadata = [{"category": "starwars"} for _ in batch_texts]
        
        collection.add(
            ids=batch_ids,
            documents=batch_texts,
            metadatas=batch_metadata,
            embeddings=batch_embeddings
        )
        print(f"✓ Added batch {i//BATCH_SIZE + 1}/{(len(all_text_passages)-1)//BATCH_SIZE + 1}")
    
    print(f"✅ Added {len(all_text_passages)} passages to Chroma")

vector_db = Chroma(collection_name="starwars", client=client)
retriever = vector_db.as_retriever(search_kwargs={"k": 5})  # Increased to 5 for better context

# ----------------------
# 3️⃣ Gemini LLM + RAG Zinciri
# ----------------------
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

template = """
You are a wise Jedi historian and Star Wars lore expert, residing in the Jedi Archives on Coruscant.
Your duty is to answer questions from travelers across the galaxy using only the knowledge preserved in the Archives, provided in the context below.

You must never fabricate or invent information beyond what is written in the Archives.
If a detail is missing or uncertain, respond with: "It appears that the Archives hold no record of such knowledge."

Provide your responses in a detailed, lore-accurate, and immersive manner, consistent with the tone of the Star Wars universe.

---
📜 Context (from the Archives):
{context}

🛰️ Question from the Traveler:
{question}

💫 Jedi Historian's Answer:
"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | RunnableLambda(lambda x: str(llm.invoke(x.to_string())))
)

chat_history = []

def ask_question(question):
    try:
        # Get response from RAG chain
        answer = rag_chain.invoke(question)
        
        # Handle different response formats
        if isinstance(answer, dict):
            answer = answer.get("content", str(answer))
        answer = str(answer)
        
        # Debug print
        print(f"Q: {question}")
        print(f"A: {answer}")
        
        # Update chat history
        chat_history.append([question, answer])
        return chat_history, ""
    except Exception as e:
        error_msg = f"The Archives are experiencing technical difficulties: {str(e)}"
        print(f"Error: {error_msg}")
        chat_history.append([question, error_msg])
        return chat_history, ""



# ----------------------
# 4️⃣ Gradio arayüzü
# ----------------------
with gr.Blocks(theme=gr.themes.Soft(), title="Star Wars RAG Chatbot") as demo:
    gr.Markdown("""
    # 🌌 Star Wars RAG Chatbot
    Ask any question about the Star Wars universe and get answers from our knowledge base!
    """)

    #chatbot = gr.Chatbot(label="Star Wars Expert", height=400, type="messages")
    chatbot = gr.Chatbot(label="Star Wars Expert", height=400)

    txt = gr.Textbox(placeholder="Ask me anything about Star Wars...", container=False)
    submit_btn = gr.Button("Send")

    txt.submit(ask_question, inputs=[txt], outputs=[chatbot, txt])
    submit_btn.click(ask_question, inputs=[txt], outputs=[chatbot, txt])

if __name__ == "__main__":
    # Hugging Face Spaces configuration
    demo.queue(max_size=20)  # Allow multiple requests in queue
    demo.launch(
        server_name="0.0.0.0",  # Required for Spaces
        server_port=7860,
        share=True,
        max_threads=4,  # Limit concurrent processing
        show_error=True  # Show detailed error messages
    )


