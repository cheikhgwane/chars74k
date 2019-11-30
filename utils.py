import os 
import shutil

x = 1

class Loader:
    
    def __init__(self,path):
        self.path =  path
    
    def loadData():
        if(not(os.path.exist(path))):
               raise("Directory doesn't exist")
 
        dnames = os.listdir(path)
        data = {}
        for _d in dnames:
               print(_d)
            
        
        
def hello():
    print("Hello world")
  
