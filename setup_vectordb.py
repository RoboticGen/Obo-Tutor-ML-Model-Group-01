import os
import sys
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema.document import Document

load_dotenv()

print("Setting up embedding model...")
# Initialize the embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Creating sample documents...")
# Create some sample documents about a project proposal
sample_texts = [
    "This is a project proposal for developing a robotic tutoring system. The system aims to provide personalized instruction and adaptive teaching to students.",
    "The robotic tutor will use artificial intelligence to understand student needs and provide customized learning experiences.",
    "Key features include natural language processing, computer vision for gesture recognition, and machine learning for adaptive learning paths.",
    "The project involves developing both hardware and software components for an interactive educational robot.",
    "Technical specifications include sensors for student interaction, processing units for AI computations, and educational content delivery systems.",
    "The system requirements specify that the robot should be able to teach various subjects including mathematics, science, and programming.",
    "Implementation phases include design, prototyping, testing, and deployment in educational environments.",
    "Expected outcomes include improved student engagement, personalized learning experiences, and enhanced educational effectiveness."
]

documents = []
for i, text in enumerate(sample_texts):
    doc = Document(
        page_content=text,
        metadata={
            'id': f'doc_{i}',
            'type': 'text',
            'source': 'project_proposal'
        }
    )
    documents.append(doc)

print(f"Creating vector store with {len(documents)} documents...")

# Create vector store
vectorstore = Chroma.from_documents(
    documents=documents, 
    embedding=embedding_model, 
    persist_directory=os.getenv("VECTOR_DB_DIR")
)

print(f"Vector store created successfully!")
print(f"Total documents in vector store: {vectorstore._collection.count()}")

# Test search
print("\nTesting search functionality...")
queries = [
    "what is this document about",
    "what is the robotic tutor",
    "what are the key features",
    "technical specifications"
]

for query in queries:
    print(f"\nQuery: '{query}'")
    relevant_docs = vectorstore.similarity_search(query, k=2)
    for i, doc in enumerate(relevant_docs):
        print(f"  Result {i+1}: {doc.page_content[:100]}...")

print("\nVector database setup complete!")