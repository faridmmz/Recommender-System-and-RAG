# Recommender-System-and-RAG

**A gaming-focused, LLM-powered recommendation engine with real-time retrieval from ElasticSearch.**  

This project combines a **content-based recommender system** with a **Retrieval-Augmented Generation (RAG)** assistant, purpose-built for the gaming industry.  
The dataset consists of **self-crawled product data from Amazon’s gaming category**, covering **video games, consoles, and gaming accessories**.  

The system indexes this data in **ElasticSearch**, and the RAG assistant — powered by the **`TheBloke/Llama-2-7B-Chat-GPTQ`** large language model — interprets user queries, retrieves the most relevant product details, and generates context-aware, gaming-specific recommendations.

By combining **fast search** with **generative reasoning**, the system ensures every suggestion is **accurate, up-to-date, and grounded in real product data**.

---

## 🚀 Key Features

- **Domain-Specific Dataset**  
  Self-crawled product data exclusively from Amazon’s gaming category.

- **Gaming-Tailored RAG Assistant**  
  Specializes in recommending **games, consoles, and accessories** based on user preferences.

- **ElasticSearch-Powered Retrieval**  
  Cloud-hosted ElasticSearch for scalable, low-latency product search.

- **LLM Reasoning with Domain Expertise**  
  The large language model doesn’t just retrieve — it explains and tailors recommendations for gamers.

- **Content-Based Recommendations**  
  Suggests similar items using vector embeddings and semantic similarity.

- **Natural Language Shopping Experience**  
  Users can ask conversational questions and receive well-reasoned, gamer-friendly responses.

---

## 🧠 Why the LLM Matters

Gaming products aren’t one-size-fits-all — preferences depend on **platform, play style, budget, and performance needs**.  
Here, the **LLM** acts as a **gaming domain expert** that:

- Understands **free-form gamer queries** (“Recommend a PS5 headset with 3D audio under $150”)  
- Retrieves **only relevant gaming products** from ElasticSearch  
- Generates **friendly, helpful answers** that gamers can act on immediately  
- Tailors results to **gaming-specific criteria** like latency, FPS optimization, or controller ergonomics

This transforms the system into more than just a search tool — it’s a **personal gaming shopping assistant**.

---

## 🧩 LLM Prompting Strategy

The assistant is primed with a **system prompt** to enforce consistency and expertise:

SYSTEM: You are a helpful, respectful and honest assistant and expert of game-related products. You know that your goal is to provide accurate, useful, and friendly recommendations for gamers, whether they are looking for hardware, consoles, accessories, or games.


This ensures responses are **gaming-focused** and aligned with the needs of the target audience.

---

## 🛠 Tech Stack

- **Languages:** Python  
- **Search Engine:** ElasticSearch (Cloud Deployment)  
- **LLM:** [`TheBloke/Llama-2-7B-Chat-GPTQ`](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GPTQ)  
- **LLM & NLP Frameworks:**  
  - HuggingFace Transformers  
  - Auto-GPTQ for quantized model loading  
  - SentencePiece for tokenizer handling  
  - Optimum & Accelerate for optimized inference  
- **Data Processing:** Pandas, NumPy  
- **Model Integration:** Retrieval-Augmented Generation pipeline

---

## ⚙️ How It Works

1. **Data Crawling**  
   - Scraped Amazon’s gaming category (games, consoles, accessories)  
   - Collected titles, descriptions, pricing, ratings, and features

2. **Embedding & Indexing**  
   - Generated vector embeddings for all product descriptions  
   - Stored embeddings in ElasticSearch for semantic search

3. **Query Processing**  
   - User query is passed to the LLM with the **gaming system prompt**  
   - LLM formulates a retrieval query for ElasticSearch

4. **Retrieval-Augmented Generation**  
   - Fetches top relevant products  
   - LLM generates grounded, gaming-specific recommendations

5. **Recommendation Delivery**  
   - Provides product names, descriptions, and reasoning for suggestions

---

## 💬 Example Query & Response

**User:**  
> "Recommend an Xbox controller for competitive FPS gaming under $100."

**Assistant:**  
> "Here are three great options:  
> 1. **Xbox Elite Series 2 Core** – Adjustable-tension thumbsticks, hair-trigger locks  
> 2. **Razer Wolverine V2 Chroma** – Mecha-tactile buttons, remappable functions  
> 3. **PowerA Fusion Pro 2** – 4 programmable paddles, high comfort grip  
> All are priced under $100 and optimized for low-latency competitive play.  
> Would you like me to compare build quality and durability?"

---

## 📈 Future Improvements

- Expand dataset to include **gaming laptops and streaming gear**  
- Add **price tracking and deal alerts**  
- Integrate **multimodal search** with product images  
- Fine-tune LLM on **gaming review text** for even better reasoning  
