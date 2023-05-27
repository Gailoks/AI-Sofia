from io import StringIO

class StringBuilder:
    string = None
 
    def __init__(self):
        self.string = StringIO()
 
    def append(self, str:str):
        self.string.write(str)
 
    def build(self):
        return self.string.getvalue()