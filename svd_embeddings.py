import numpy as np
import pickle

class Embeddings():

    def __init__(self, path_embed='svd_ppmi_embeddings_50dim.npy', path_vocab='svd_ppmi_embeddings_vocab.pkl'):
        self.wi = self._loadVocab(path_vocab)
        self.wordvec = self._loadEmbed(path_embed)


    def _loadEmbed(self, path):
        return np.load(path)


    def _loadVocab(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data


    def cossim(self, w1, w2):
        return self.getWordvec(w1).dot(self.getWordvec(w2))


    def getWordvec(self, w):
        if isinstance(w, str):
            w = self.wi[w]
        return self.wordvec[w]
