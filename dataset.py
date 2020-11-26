import pickle
from torch.utils.data import Dataset
from torch import tensor
import spacy

class YelpDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        self.nlp = spacy.load('en_core_web_md')
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        stars, tokens = self.data[idx]
        word_vecs = list(map(lambda word: self.nlp(word).vector, tokens))
        return stars, word_vecs
