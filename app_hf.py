import os
import json
from pathlib import Path
import gradio as gr
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from chromadb import Client, Settings
from langchain.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
import time

# Constants and Settings
CHROMA_DIR = Path("chroma_db")
CHROMA_DIR.mkdir(exist_ok=True)

# Load environment variables from Hugging Face Spaces
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY in your Hugging Face Space settings")

# Initialize Chroma with persistent storage and optimized settings
client = Client(Settings(
    persist_directory=str(CHROMA_DIR),
    is_persistent=True,
    anonymized_telemetry=False
))

# Get or create collection
collection = client.get_or_create_collection(
    name="starwars",
    metadata={"hnsw:space": "cosine"}
)

def initialize_rag():
    """Initialize the RAG components with error handling"""
    try:
        # Initialize vector store
        vector_db = Chroma(
            client=client,
            collection_name="starwars"
        )
        
        # Configure retriever
        retriever = vector_db.as_retriever(
            search_kwargs={
                "k": 5,
                "fetch_k": 20  # Fetch more candidates for better selection
            }
        )
        
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.7,
            convert_system_message_to_human=True
        )
        
        return retriever, llm
    except Exception as e:
        print(f"Error initializing RAG components: {e}")
        raise

# Template for the chatbot
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

# Initialize RAG components
retriever, llm = initialize_rag()

def format_docs(docs):
    """Format retrieved documents with error handling"""
    try:
        return "\n\n".join(doc.page_content for doc in docs)
    except Exception as e:
        print(f"Error formatting documents: {e}")
        return ""

# Create RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | RunnableLambda(lambda x: str(llm.invoke(x.to_string())))
)

# Initialize chat history
chat_history = []

def ask_question(question):
    """Process questions with error handling and rate limiting"""
    try:
        # Add rate limiting
        time.sleep(0.5)  # Prevent too rapid requests
        
        # Get response from RAG chain
        answer = rag_chain.invoke(question)
        if isinstance(answer, dict) and "content" in answer:
            answer = answer["content"]
            
        # Update chat history
        chat_history.append([question, answer])
        return chat_history, ""
        
    except Exception as e:
        error_msg = f"The Archives are experiencing technical difficulties: {str(e)}"
        print(f"Error: {error_msg}")
        chat_history.append([question, error_msg])
        return chat_history, ""

# Create Gradio interface with optimized settings
with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Star Wars RAG Chatbot",
    analytics_enabled=False  # Disable analytics for better performance
) as demo:
    gr.Markdown("""
    # 🌌 Star Wars RAG Chatbot
    Ask any question about the Star Wars universe and get answers from our knowledge base!
    """)

    chatbot = gr.Chatbot(
        label="Star Wars Expert",
        height=400,
        bubble_full_width=False,  # Optimize rendering
        show_copy_button=True
    )
    
    txt = gr.Textbox(
        placeholder="Ask me anything about Star Wars...",
        container=False,
        scale=8
    )
    
    submit_btn = gr.Button("Send", scale=2)

    txt.submit(ask_question, inputs=[txt], outputs=[chatbot, txt])
    submit_btn.click(ask_question, inputs=[txt], outputs=[chatbot, txt])

# Configure for Hugging Face Spaces
if __name__ == "__main__":
    demo.queue(concurrency_count=1)  # Limit concurrent requests
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        max_threads=4,
        show_error=True
    )