import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from package import (
    SimpleDirectoryReader
)
class Load_Data:
    def __init__(self,data:str):
        '''
        data: thư mục chứa các file pdf
        '''
        self.data:str = data

    def Load(self):
        return SimpleDirectoryReader(self.data).load_data()
    
