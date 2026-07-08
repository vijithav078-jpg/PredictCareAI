from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
)


# Load FAISS database
vector_db = FAISS.load_local(
    "vector_db",
    embeddings,
    allow_dangerous_deserialization=True
)

print("✅ Vector Database Loaded Successfully")


def search_manual(question):

    results = vector_db.similarity_search(question, k=5)

    recommendations = []

    keywords = [
        "recommend",
        "maintenance",
        "replace",
        "inspect",
        "check",
        "lubric",
        "coolant",
        "bearing",
        "failure",
        "warning",
        "emergency",
        "preventive",
        "troubleshooting",
        "solution"
    ]

    for doc in results:

        lines = doc.page_content.split("\n")

        for line in lines:

            line = line.strip()

            if len(line) < 20:
                continue

            lower = line.lower()

            if any(word in lower for word in keywords):

                if line not in recommendations:
                    recommendations.append("✔ " + line)

    if recommendations:
        return "\n\n".join(recommendations[:5])

    # Fallback
    return results[0].page_content