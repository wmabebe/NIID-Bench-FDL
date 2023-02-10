class Node:
    def __init__(self,id,model,data=None,max_peers=2):
        self.id = id
        self.model = model
        self.data = data
        self.neighbors = []
        self.max_peers = max_peers
        self.candidates = []
        self.encountered = []
    
    def __str__(self) -> str:
        return str(self.id)