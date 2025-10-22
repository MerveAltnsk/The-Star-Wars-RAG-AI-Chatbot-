# 🌌 Star Wars RAG Chatbot

<img src="https://images.steamusercontent.com/ugc/2258055576432666610/F333CDBF79C5E51910C86563B3D8F0D91E717B48/?imw=637&imh=358&ima=fit&impolicy=Letterbox&imcolor=%23000000&letterbox=true" width="600" />

A **Retrieval-Augmented Generation (RAG)** chatbot that answers questions about the **Star Wars universe** using structured data from SWAPI (Star Wars API). Built with **LangChain**, **Gemini**, and **Gradio**, this chatbot combines semantic search and generative AI for accurate, lore-consistent responses.  

---

## 🚀 Project Overview

The Star Wars RAG Chatbot enables users to:

- **Explore Star Wars lore**: Ask questions about characters, films, planets, starships, vehicles, and species.  
- **Semantic retrieval**: Uses vector embeddings to fetch contextually relevant passages from the knowledge base.  
- **RAG-powered responses**: Integrates Google's Gemini model for detailed and immersive answers.  
- **Interactive web interface**: Chat-style interface built with Gradio, accessible via web browser.

> Try asking questions like *"What is the height of Darth Vader?"* or *"Who directed The Empire Strikes Back?"* and get instant answers from the Jedi Archives.

---

## 🖼️ Chat Interface

<img width="1020" height="519" alt="Image" src="https://github.com/user-attachments/assets/e8b73350-0630-4839-a381-d80b84417366" />
<img width="1017" height="518" alt="Image" src="https://github.com/user-attachments/assets/958c5543-f53b-4809-87b1-5d6910e3db67" />
---

## 📓 Notebook

View the full Colab notebook with all code, outputs, and explanations:  

[Open Colab Notebook](https://colab.research.google.com/drive/1u7n4n3pa4hPlt86A5W626-POKbiSzJ3b?usp=sharing)



## Project Purpose

This chatbot demonstrates how to build a knowledge-based question-answering system that combines:

- Structured data retrieval from SWAPI
- Vector embeddings for semantic search
- RAG architecture with Google's Gemini model
- Interactive web interface with Gradio

The project serves as both a practical example of modern NLP techniques and an educational tool for Star Wars fans.

## Dataset Description

The knowledge base of this project is built primarily from the **Star Wars API (SWAPI)**, a public API providing canonical Star Wars universe information. The dataset contains structured data for the following entities:

### 1. Characters (People)
- Name, height, mass, hair color, skin color, eye color, birth year, and gender.
- Examples: Luke Skywalker, Darth Vader, Yoda, Leia Organa.
- Data captures relationships such as species, homeworld, and appearances in films.

### 2. Films
- Title, episode number, director, producer, release date.
- Example: *A New Hope*, *The Empire Strikes Back*.
- This allows the chatbot to answer production-related questions.

### 3. Species
- Name, classification, designation, average height, average lifespan, language.
- Example: Wookiees, Ewoks, Humans, Droids.

### 4. Starships
- Name, model, manufacturer, cost, length, max speed, crew, passengers, cargo capacity.
- Example: Millennium Falcon, X-wing, Star Destroyer.

### 5. Vehicles
- Name, model, manufacturer, cost, length, crew, passengers, cargo capacity.
- Example: Sand Crawler, Speeder Bikes.

### 6. Planets
- Name, rotation period, orbital period, diameter, climate, gravity, terrain, surface water, population.
- Example: Tatooine, Alderaan, Hoth, Endor.

### Key Notes
- All API responses are converted into **natural language text passages** suitable for semantic search.
- Data preserves entity relationships, e.g., which characters appear in which films or which species belong to which planets.
- The dataset allows the RAG pipeline to retrieve **contextually relevant passages** for accurate, lore-consistent answers.

### Dataset Size
- The final dataset includes **hundreds of passages**: all characters, films, species, starships, vehicles, and planets indexed with embeddings.
- Stored in **Chroma vector database** for fast retrieval using semantic similarity search.


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
Add GOOGLE_API_KEY to Hugging Face Space secrets for deployment.

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

The Hugging Face Space for this project is not yet live.  
You can run the chatbot locally following the [Local Setup](#local-setup) instructions.

## Web Interface & Usage

The chatbot provides a simple Gradio-based web interface:

1. Type your question in the text box.
2. Click the **Send** button.
3. The chatbot will display its answer instantly in the chat window.

You can access the deployed chatbot at: [https://huggingface.co/spaces/MerveAltnsk/star-wars-rag-chatbot](https://huggingface.co/spaces/MerveAltnsk/star-wars-rag-chatbot)


## Future Improvements

- Add more Star Wars data sources
- Implement conversation memory
- Add source citations to responses
- Optimize retrieval for faster responses
- Add character-specific personas

## Acknowledgements

- Data: [SWAPI](https://swapi.dev/)
- Libraries: LangChain, Gemini, SentenceTransformers, Chroma, Gradio
