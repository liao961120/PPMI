import os
import pickle
import numpy as np


class Embeddings():

    def __init__(self, embed_dim=300, path_vocab='svd_ppmi_embeddings_vocab.pkl'):
        path_embed = f'svd_ppmi_embeddings_{embed_dim}dim.npy'
        if not (os.path.exists(path_embed) and os.path.exists(path_vocab)):
            raise Exception(f"`{path_embed}` or `{path_vocab}` doesn't exist!")

        self.wi = self._loadVocab(path_vocab)
        self.wordvec = self._loadEmbed(path_embed)


    def _loadEmbed(self, path):
        return np.load(path)


    def _loadVocab(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data


    def cossim(self, w1, w2):
        """Compute cosine similarity of two words.
        
        Parameters
        ----------
        w1 : Union[str, int]
            A string or integer representing a word. Integer corresponds to
            the row index of :py:attr:`.Embeddings.wordvec`.
        w2 : Union[str, int]
            A string or integer representing a word. Integer corresponds to
            the row index of :py:attr:`.Embeddings.wordvec`.
        
        Returns
        -------
        float
            Cosine similarity of the two terms.
        """
        return self.getWordvec(w1).dot(self.getWordvec(w2))


    def getWordvec(self, w):
        """Get word vector representation.
        
        Parameters
        ----------
        w : Union[str, int]
            A string or integer representing a word. Integer corresponds to
            the row index of :py:attr:`.Embeddings.wordvec`.
        
        Returns
        -------
        numpy.ndarray
            A 1d array of floats representing a word vector.
        """
        if isinstance(w, str):
            w = self.wi[w]
        return self.wordvec[w]
