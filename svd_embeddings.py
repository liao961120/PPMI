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


    def analogy(self, a1, a2, b1, e=0.001, top_n=3):
        """Get analogy by cosine multiplication
        
        Parameters
        ----------
        a1 : Union[str, int]
            A word
        a2 : Union[str, int]
            A word
        b1 : Union[str, int]
            A word
        top_n : int, optional
            Number of analogies to return, by default 3
        """

        # Get vocab
        vocab = set(self.wi.keys())
        for w in [a1, a2, b1]:
            if w in vocab:
                vocab.remove(w)

        # Analogy cosine (see Levy et al. 2015)
        def cosMul(b2):
            return (self.cossim(b2, a1) * self.cossim(b2, b1)) / (self.cossim(b2, a1) + e)

        similarities = sorted(( (cosMul(b2), b2) for b2 in vocab ), reverse=True)

        return similarities[:top_n]



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
