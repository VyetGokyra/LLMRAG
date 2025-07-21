from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_chroma import Chroma
from document_loader import split_docs
from llm import llm
import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Initialize SentenceTransformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

class ChromaEmbeddingFunction:
    def __init__(self, model):
        self.model = model
        
    def __call__(self, input):
        if isinstance(input, str):
            input = [input]
        return self.model.encode(input, normalize_embeddings=True).tolist()
    
    def embed_documents(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, normalize_embeddings=True).tolist()
    
    def embed_query(self, text):
        return self.model.encode(text, normalize_embeddings=True).tolist()
    
embedding_function = ChromaEmbeddingFunction(model)
# Ensure the persistence directory exists
PERSIST_DIRECTORY = os.path.join(os.path.dirname(__file__), 'chroma_db')
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

try:
    print(f"Number of documents to process: {len(split_docs)}")
    print(f"Processing documents in batches...")
    
    # Use the previously initialized embedding_function
    
    # Initialize Chroma client first
    client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)
    
    # Create or get collection
    collection_name = "document_collection"
    try:
        collection = client.get_collection(collection_name)
        print(f"Found existing collection: {collection_name}")
    except:
        collection = client.create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
        print(f"Created new collection: {collection_name}")
    
    # Process documents in smaller batches
    batch_size = 50  # Reduced batch size to handle fewer tokens per request
    total_batches = (len(split_docs) + batch_size - 1) // batch_size
    
    for i in range(0, len(split_docs), batch_size):
        batch = split_docs[i:i + batch_size]
        batch_num = i // batch_size + 1
        print(f"Processing batch {batch_num}/{total_batches}")
        
        # Extract text and metadata from documents
        texts = [doc.page_content for doc in batch]
        metadatas = [doc.metadata for doc in batch]
        ids = [f"doc_{i+j}" for j in range(len(batch))]
        
        try:
            # Add documents to collection
            collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Successfully added batch {batch_num}")
        except Exception as e:
            print(f"Error processing batch {batch_num}: {str(e)}")
            continue
    
    # Create LangChain vectorstore from the collection
    vectorstore = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embedding_function
    )
    
    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
    print("Vectorstore created successfully!")
except Exception as e:
    print(f"Error creating vectorstore: {str(e)}")
    raise


contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)