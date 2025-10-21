# Star Wars RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about the Star Wars universe using data from SWAPI (Star Wars API). Built with LangChain, Gemini, and Gradio.

## Project Purpose

This chatbot demonstrates how to build a knowledge-based question-answering system that combines:

- Structured data retrieval from SWAPI
- Vector embeddings for semantic search
- RAG architecture with Google's Gemini model
- Interactive web interface with Gradio

The project serves as both a practical example of modern NLP techniques and an educational tool for Star Wars fans.

## Dataset Description

The knowledge base is built from the Star Wars API (SWAPI), which provides canonical information about:

- Characters (people)
- Films
- Species
- Vehicles
- Starships
- Planets

Each entity is processed into text passages that preserve relationships and context, making them suitable for retrieval and question answering.

## Methods

The project implements a RAG pipeline with the following components:

1. Data Collection & Processing

   - Fetches data from SWAPI endpoints
   - Converts structured API responses to natural text passages
   - Preserves relationships between entities

2. Embeddings & Vector Store

   - Uses SentenceTransformers (all-MiniLM-L6-v2) for embeddings
   - Stores vectors in a Chroma database
   - Enables semantic similarity search

3. RAG Implementation

   - LangChain orchestrates the retrieval pipeline
   - Top-k relevant passages retrieved for each query
   - Context-aware responses using Gemini model

4. User Interface
   - Gradio web interface
   - Chat-style interaction
   - Real-time response generation

## Results

The chatbot successfully:

- Processes and indexes the complete SWAPI dataset
- Provides accurate answers to Star Wars-related questions
- Maintains conversation context
- Delivers responses with an engaging Jedi historian persona

Performance metrics:

- Response time: 2-5 seconds average
- Accurate information retrieval from the knowledge base
- Natural, contextual answers that stay true to Star Wars lore

## Example Questions

Here are some example questions you can ask the chatbot, organized by category. These questions demonstrate the range of information available through the SWAPI dataset:

### Characters and Biography

- "What is the height of Darth Vader?"
- "How much does Chewbacca weigh?"
- "What is Leia Organa's birth year?"
- "What color is Han Solo's hair?"
- "What gender is Yoda?"

### Films and Production

- "Who directed The Empire Strikes Back?"
- "When was A New Hope released?"
- "What is the episode number of Revenge of the Sith?"
- "Who produced Return of the Jedi?"

### Planets and Locations

- "What is the climate of Tatooine?"
- "How many residents does Alderaan have?"
- "What is the terrain of Hoth?"

### Ships and Technology

- "What is the model of the Millennium Falcon?"
- "Who pilots X-wing starships?"
- "What is the maximum speed of Star Destroyer?"

### Species and Culture

- "What is the average lifespan of Wookiees?"
- "What language do Ewoks speak?"
- "What is the classification of Humans in Star Wars?"

Each question is answered using information from the SWAPI database, enhanced with the chatbot's Jedi historian persona for an immersive experience.

## Local Setup

1. Clone the repository

```bash
git clone https://github.com/MerveAltnsk/StarWarsRAGChatbot.git
cd StarWarsRAGChatbot
```

2. Create and activate virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# OR
source .venv/bin/activate  # Linux/Mac
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Set up environment variables

```bash
cp .env.template .env
# Edit .env and add your GOOGLE_API_KEY
```

5. Run the application

```bash
python app.py
```

The web interface will be available at http://localhost:7860

## Deployment

The chatbot can be deployed on Hugging Face Spaces or any platform supporting Gradio apps.

1. Create a new Space on Hugging Face
2. Upload project files
3. Add GOOGLE_API_KEY to Space secrets
4. Deploy and access via provided URL

## Deploy Link

Visit the live demo: https://huggingface.co/spaces/MerveAltnsk/star-wars-rag-chatbot

## Future Improvements

- Add more Star Wars data sources
- Implement conversation memory
- Add source citations to responses
- Optimize retrieval for faster responses
- Add character-specific personas

## Acknowledgements

- Data: [SWAPI](https://swapi.dev/)
- Libraries: LangChain, Gemini, SentenceTransformers, Chroma, Gradio
