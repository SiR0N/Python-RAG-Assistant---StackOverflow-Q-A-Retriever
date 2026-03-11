# Python RAG Assistant - StackOverflow Q&A Retriever
This project implements a Retrieval-Augmented Generation (RAG) system that answers Python-related questions using real StackOverflow data

- Sentence-transformer embeddings
- FAISS vector search
- A cleaned and filtered StackOverflow dataset
- A Gemini generative model to produce grounded answers
The assistant retrieves the most relevant Q&A pairs and generates a final answer strictly based on the retrieved context.

Features
- Loads and preprocesses the StackOverflow Python Questions dataset from Kaggle.
- Filters answers by keeping only those with score above the question’s average.
- Builds rich documents combining:
  - question title
  - question body
  - question score
  - top‑scoring answers
- Cleans HTML using BeautifulSoup.
- Generates embeddings using:
  - all-MiniLM-L6-v2
- Indexes documents with FAISS (IndexFlatL2).
- Retrieves the most relevant documents for any query.
- Uses Gemini to generate a final answer grounded in the retrieved context.

📦 Installation
1. Clone the repository
git clone https://github.com/SiR0N/python-rag-assistant.git


2. Install dependencies
pip install -r requirements.txt


3. Create a .env file
The notebook loads your API key using load_dotenv():
openai_key=YOUR_API_KEY



🧠 RAG Pipeline Overview
1. Dataset loading
The notebook downloads the StackOverflow Python dataset:
path = kagglehub.dataset_download("stackoverflow/pythonquestions")


It loads:
- Questions.csv
- Answers.csv
- Tags.csv

2. Data filtering
- Keep only questions with positive score.
- Compute the mean score of answers per question.
- Keep only answers with score greater than the mean.

3. Document construction
Each document includes:
- Title
- Question body
- Question score
- A formatted list of top answers
Example from the notebook:
“Title: Find broken symlinks with Python
Question Score: 13
Answer 1 - Score: 11 …”

4. HTML cleaning
BeautifulSoup(text, "html.parser").get_text()


5. Embedding generation
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(documents, normalize_embeddings=True)


6. Vector index with FAISS
index = faiss.IndexFlatL2(dim)
index.add(embeddings)


7. Retrieval
distances, indices = index.search(q_emb, k)


8. Answer generation with Gemini
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)



🧪 Example Usage
result = rag_answer("How do I open a CSV in Python?")
print(result["answer"])


Example output:
“You can open a CSV file in Python using the open() function…
or with with open('file.csv', 'rb') as csvfile: …”


📈 Future Improvements
- Add reranking (Cohere, Jina, Voyage).
- Implement context compression.
- Serve the RAG pipeline via FastAPI.
- Build a UI with Streamlit or Gradio.

📜 License
MIT License.
