import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from package import (
    SentenceSplitter
)

class Chunking:
    def __init__(self,documents):
        '''
        documents: tài liệu khi mà mình load xong
        '''
        self.documents = documents
    
    def Chunk_data(self):
        text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=128)
        return text_splitter(self.documents)
