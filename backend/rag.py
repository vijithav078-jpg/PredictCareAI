from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# PDF Path
PDF_PATH = "rag_docs/predictcareAI_manual.pdf"

# Load PDF
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

# Split PDF into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)

# Create Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create FAISS Vector Database
vector_db = FAISS.from_documents(docs, embeddings)

print("✅ PDF Loaded Successfully")
print("✅ Vector Database Created")


def search_manual(question):

    results = vector_db.similarity_search(question, k=2)

    recommendations = []

    ignore = [
        "maintenance",
        "maintenance activity",
        "frequency",
        "table of contents",
        "predictcare ai",
        "page"
    ]

    for doc in results:

        for line in doc.page_content.split("\n"):

            line = line.strip()

            if len(line) < 15:
                continue

            if line.lower() in ignore:
                continue

            if line not in recommendations:
                recommendations.append(line)

    answer = ""

    for item in recommendations[:5]:
        answer += f"✔ {item}\n\n"

    return answer