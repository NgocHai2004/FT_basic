import os
class Load_Data:
    def __init__(self,path_folder:str):
        '''
        path_folder: thư mục chứa các file pdf
        '''
        self.path_folder:str = path_folder

    def load(self):
        return Load_Data(self.path_folder).Load()
    
