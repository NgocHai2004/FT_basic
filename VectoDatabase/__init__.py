import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from package import (
    faiss,
    FaissVectorStore,
    StorageContext,
    VectorStoreIndex
)

class VectorDatabase:
    def __init__(self, embed_model, dimension=384):
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.vector_store = FaissVectorStore(faiss_index=self.faiss_index)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.embed_model = embed_model

    def build_index(self, documents):
        self.index = VectorStoreIndex.from_documents(
            documents, 
            storage_context=self.storage_context,
            embed_model=self.embed_model
        )

    def get_retriever(self, top_k=3):
        return self.index.as_retriever(similarity_top_k=top_k)