# Python RAG Assistant — StackOverflow Q&A Retriever

This project implements a Retrieval-Augmented Generation (RAG) system that answers Python programming questions using real StackOverflow data.

It combines:

- Sentence-transformer embeddings
- FAISS vector search
- A cleaned and curated StackOverflow dataset
- Gemini for grounded answer generation

The assistant retrieves the most relevant Q&A pairs and produces a final answer strictly based on the retrieved context.

---

## 1. Introduction

This project demonstrates how to build a complete Retrieval-Augmented Generation (RAG) pipeline using Python, StackOverflow data, FAISS vector search, sentence-transformer embeddings, and Gemini as the LLM.  
It loads real StackOverflow questions and answers, cleans and filters them, embeds them, indexes them, retrieves the most relevant ones, and generates grounded answers.

---

## 2. Features

- Loads and preprocesses the StackOverflow Python Questions dataset from Kaggle.
- Filters answers by keeping only those with score above the question’s average.
- Builds rich documents containing:
  - Question title  
  - Question body  
  - Question score  
  - Top-scoring answers  
- Cleans HTML using BeautifulSoup.
- Generates embeddings using:
  - `all-MiniLM-L6-v2`
- Indexes documents with FAISS (`IndexFlatL2`).
- Retrieves the most relevant documents for any query.
- Uses Gemini to generate a final answer grounded in the retrieved context.

---

## 3. Project Structure

```text
Python-RAG-Assistant---StackOverflow-Q-A-Retriever
├── README.md
├── Python_RAG_Assistant.ipynb
├── requirements.txt
```


---

## 4. Installation

### 4.1 Clone the repository

    git clone https://github.com/SiR0N/Python-RAG-Assistant---StackOverflow-Q-A-Retriever.git
    cd Python-RAG-Assistant---StackOverflow-Q-A-Retriever

### 4.2 Install dependencies

    pip install -r requirements.txt

### 4.3 Create a `.env` file

    ai_key=YOUR_API_KEY

---

## 5. RAG Pipeline Overview

### 5.1 Dataset Loading

    import kagglehub
    path = kagglehub.dataset_download("stackoverflow/pythonquestions")

Loads:

- Questions.csv
- Answers.csv
- Tags.csv

---

### 5.2 Data Filtering

This step filters the dataset to keep only high‑quality questions and answers:

- Only questions with a positive score are kept.
- The mean score of answers per question is computed.
- Only answers with a score above the mean for their question are kept.
---
    questions = questions[questions["Score"] > 0]

    mean_scores = answers.groupby("ParentId")["Score"].mean().reset_index()
    mean_scores.rename(columns={"Score": "mean_score"}, inplace=True)

    answers_with_mean = answers.merge(mean_scores, on="ParentId", how="left")
    answers_filtered = answers_with_mean[
        answers_with_mean["Score"] > answers_with_mean["mean_score"]
    ]

---

### 5.3 Document Construction

This step builds a single text document per question, combining:

- Title  
- Body  
- Score  
- All selected answers  
---
    def build_document(row):
        question_score = row.get("Score", "")
        answers = row.get("answers_list") or []

        formatted_answers = []
        for i, ans in enumerate(answers, start=1):
            formatted_answers.append(
                f"Answer {i} - Score: {ans['score']}\n{ans['body']}"
            )

        answers_text = "\n\n".join(formatted_answers)

        return (
            f"Title: {row.get('Title', '')}\n"
            f"Question Score: {question_score}\n\n"
            f"Question:\n{row.get('Body', '')}\n\n"
            f"Answers:\n{answers_text}"
        )

    df["document"] = df.apply(build_document, axis=1)

---

### 5.4 HTML Cleaning

StackOverflow content contains HTML.  
This step removes all tags and keeps only plain text.

    from bs4 import BeautifulSoup

    def clean(text):
        return BeautifulSoup(str(text), "html.parser").get_text()

    df["document"] = df["document"].apply(clean)

---

### 5.5 Fix Encoding Issues and Save

Some entries contain invalid Unicode characters.  
This step fixes them and saves the cleaned dataset.

    def fix_surrogates(text):
        if isinstance(text, str):
            return text.encode("utf-8", "surrogatepass").decode("utf-8", "ignore")
        return text

    df = df.applymap(fix_surrogates)

    df_clean_path = "df_clean.parquet"
    df.to_parquet(df_clean_path)

    df = pd.read_parquet(df_clean_path)

---

### 5.6 Embedding Generation

The documents are converted into vector embeddings using a sentence-transformer model.

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")

    documents = df["document"].tolist()

    embeddings = model.encode(
        documents,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=512,
        show_progress_bar=True,
    ).astype("float32")

    dim = embeddings.shape[1]

---

### 5.7 Vector Index with FAISS

FAISS is used to index the embeddings and enable fast similarity search.

    import faiss

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

---

### 5.8 Retrieval

Given a user query, this function retrieves the most relevant documents.

    def retrieve(query, k=5):
        q_emb = model.encode([query], normalize_embeddings=True).astype("float32")
        distances, indices = index.search(q_emb, k)
        return df.loc[indices[0], ["document"]]

---

### 5.9 Build Context

The retrieved documents are concatenated into a single context block for the LLM.

    def build_context(rows):
        return "\n\n---\n\n".join(rows["document"].tolist())

---

### 5.10 Gemini Answer Generation

This function sends the prompt and context to Gemini and returns the generated answer.

    from google import genai

    client = genai.Client(api_key="YOUR_API_KEY")

    def call_gemini(prompt):
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return response.text

---

### 5.11 Full RAG Function

This is the complete RAG pipeline: retrieve → build context → generate answer.

    def rag_answer(query):
        retrieved = retrieve(query)
        context = build_context(retrieved)

        prompt = f"""
        You are an expert Python assistant. Answer ONLY using the information in the context.

        CONTEXT START
        {context}
        CONTEXT END

        QUESTION:
        {query}

        FINAL ANSWER (in English):
        """

        answer = call_gemini(prompt)

        return {
            "prompt": prompt,
            "retrieved_docs": retrieved,
            "answer": answer,
        }

---

## 6. Usage Example

    result = rag_answer("How do I open a CSV in Python?")
    print(result["answer"])

---


Example output:

> You can open a CSV file in Python using the `open()` function or with  
> `with open('file.csv', 'rb') as csvfile:` and then use the `csv` module to read its contents.

---

## 7. Requirements

    faiss-cpu
    sentence-transformers
    beautifulsoup4
    pandas
    numpy
    python-dotenv
    google-genai
    kagglehub
    pyarrow

Install with:

    pip install -r requirements.txt

---


## 8. License

MIT License.



