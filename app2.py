import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import torch
from package import RAGPipeline
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", device=device)

pipeline = RAGPipeline(
    data_folder="Data",
    embed_model=embed_model,
    api_key="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",
    model_name="meta-llama/llama-4-scout-17b-16e-instruct"
)

question = "Mamba là gì?"
answer = pipeline.run(question)
print(answer)
