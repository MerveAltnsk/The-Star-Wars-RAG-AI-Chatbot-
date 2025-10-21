import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from chromadb import Client
from langchain.vectorstores import Chroma
import gradio as gr

# Load environment variables
load_dotenv()

# Initialize Chroma client
client = Client()
collection = client.get_collection("starwars")

# Initialize vector store
vector_db = Chroma(
    collection_name="starwars",
    client=client
)
retriever = vector_db.as_retriever(search_kwargs={"k": 5})

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Define prompt template
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

# Helper function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | RunnableLambda(lambda x: str(llm.invoke(x.to_string())))
)

# Chat history for conversation
chat_history = []

def ask_question(question):
    """Process a question and return the response with updated chat history"""
    answer = rag_chain.invoke(question)
    if isinstance(answer, dict) and "content" in answer:
        answer = answer["content"]

    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": answer})

    return chat_history, ""

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft(), title="Star Wars RAG Chatbot") as demo:
    gr.Markdown("""
    # 🌌 Star Wars RAG Chatbot
    Ask any question about the Star Wars universe and get answers from our knowledge base!
    """)

    chatbot = gr.Chatbot(label="Star Wars Expert", height=400, type="messages")
    txt = gr.Textbox(placeholder="Ask me anything about Star Wars...", container=False)
    submit_btn = gr.Button("Send")

    txt.submit(ask_question, inputs=[txt], outputs=[chatbot, txt])
    submit_btn.click(ask_question, inputs=[txt], outputs=[chatbot, txt])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)