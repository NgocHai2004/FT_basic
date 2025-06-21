import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from package import (
    HuggingFaceEmbedding,
    SemanticSplitterNodeParser
)

class Convert:
    def __init__(self,model_embedding):
        '''
        model_embedding: tên model mà mình dùng để embedding text
        '''
        self.model_embedding = model_embedding
    
    def model_embedd(self):
        return SemanticSplitterNodeParser(
    buffer_size = 1, breakpoint_percentile_threshold = 95,embed_model = self.model_embedding
)