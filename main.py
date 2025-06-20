import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from groq import Groq
import faiss
import torch

documents = SimpleDirectoryReader('Data').load_data()

text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=128)
processed_documents = text_splitter(documents)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", device=device)
splitter = SemanticSplitterNodeParser(
    buffer_size = 1, breakpoint_percentile_threshold = 95,embed_model = embed_model
)
nodes = splitter.get_nodes_from_documents(documents)
d = 384  # dimension of your embeddings (adjust based on your embed_model)
faiss_index = faiss.IndexFlatL2(d)

# Create vector store
vector_store = FaissVectorStore(faiss_index=faiss_index)

# Create storage context
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Create index from documents
index = VectorStoreIndex.from_documents(
    documents,  # your documents
    storage_context=storage_context,
    embed_model=embed_model  # your embedding model
)
retriever = index.as_retriever(similarity_top_k=3)
nodes = retriever.retrieve("Mamba là gì?")
datatext = []


cnt = 0
# Lấy text từ mỗi node
for node in nodes:
    text_tmp = node.text
    text_tmp = text_tmp.replace("\n","")
    
    # print(type(text_tmp))
    datatext.append(text_tmp)
client = Groq(api_key="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")

class Answer_Question_From_Documents:
    def __init__(self,question : str, documents : list[str]) -> None:
        self.question = question
        self.documents = documents

    def run(self) :
        context = "\n".join(self.documents)  # Tạo chuỗi context bên ngoài f-string
        input_text = f"""Question: {self.question}
        Context: {context}
        Answer:"""
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", 
                 "content": "Bạn là một trợ lý AI bạn hãy giúp tôi lọc bỏ đi nội dung không liên quan đến câu hỏi và dựa vào câu hỏi và nội dung đi theo hãy tạo ra câu trả lời đầy đủ."
                },
                {
                    "role": "user",
                    "content": self.question
                },
                {
                    "role": "assistant",
                    "content": input_text
                }
            ],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )
        return completion.choices[0].message.content
question = "Mamba là gì?"
respone = Answer_Question_From_Documents(question, datatext).run()
print(respone)