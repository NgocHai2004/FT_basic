import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from package import (
    SentenceSplitter
)

class Convert:
    def __init__(self, embed_model):
        '''
        embed_model : model để embedding
        '''
        self.embed_model = embed_model
        self.node_parser = SentenceSplitter()

    def embed_documents(self, documents):
        return self.node_parser.get_nodes_from_documents(documents)
    