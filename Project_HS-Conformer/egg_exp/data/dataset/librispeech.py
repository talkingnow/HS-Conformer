import os

class Libri100h:
    def __init__(self, path):
        self.train_set = []
        
        # train_set
        for root, _, files in os.walk(path):
            for file in files:
                if '.flac' in file:
                    self.train_set.append(os.path.join(root, file))
            