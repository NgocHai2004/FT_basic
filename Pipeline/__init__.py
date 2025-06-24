import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from package import (
    Load_Data,
    Convert,
    VectorDatabase,
    QuestionAnswering
)


class RAGPipeline:
    def __init__(self, data_folder, embed_model, api_key, model_name):
        self.data_loader = Load_Data(data_folder)
        self.text_embedder = Convert(embed_model)
        self.vector_db = VectorDatabase(embed_model)
        self.qa = QuestionAnswering(api_key, model_name)

    def run(self, question):
        documents = self.data_loader.load()
        self.vector_db.build_index(documents)
        retriever = self.vector_db.get_retriever()
        nodes = retriever.retrieve(question)
        texts = [node.text.replace("\n", "") for node in nodes]
        return self.qa.answer(question, texts)
