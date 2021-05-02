#%%
RAW_CORP = 'asbc5_plain_text.txt' #'asbc5_plain_text.txt' #'asbc_lite.txt'
WORD_FQ_THRESHHOLD = 20
WINDOW = 4  # left & right context window size
EMBEDDING_DIM = 50

import math
import pickle
import tqdm
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import svds
from time import time
from yf import beep

s = time()  # Record start time

#-------------------- Preprocess raw corpus -----------------#
with open(RAW_CORP) as f:
    corp = f.read()
corp = [ tk for text in corp.split('\n') for tk in text.split('\u3000') ]

#%%
# Get word occurence stat
vocab = set(corp)
vocab_counts = {w:0 for i, w in enumerate(vocab)}
for w in corp:
    vocab_counts[w] += 1
vocab = set( w for w, c in vocab_counts.items() if c > WORD_FQ_THRESHHOLD )

# Index corpus
wi = {tk:i for i, tk in enumerate(vocab)}
#iw ={v:k for k, v in wi.items()}

# Save vocabulary
with open("svd_ppmi_embeddings_vocab.pkl", "wb") as f:
    pickle.dump(wi, f)

vocab_size = len(vocab)


#------------------ Construct PPMI Matrix -------------------#
D = dict()
context = {i:0 for i, tk in enumerate(vocab)}
target = {i:0 for i, tk in enumerate(vocab)}

# Count co-occurrences of all (target, context) pairs
for i, tgt_w in tqdm.tqdm(enumerate(corp)):
    
    # Record cooccurences in context window
    cntx = [ w for w in corp[(i-WINDOW):i] + corp[(i+1):(i+WINDOW+1)] ]
    for cntx_w in cntx:
        
        # Filter out words with low fq
        if (tgt_w not in vocab) or (cntx_w not in vocab):
            continue

        pair = (wi[tgt_w], wi[cntx_w])

        # Update coocurrence
        if pair in D.keys():
            D[pair] += 1
        else:
            D[pair] = 1
        
        # Update target, context word count
        target[pair[0]] += 1
        context[pair[1]] += 1


# Compute PMI value
D_size = len(D)
def pmi(tgt_w, cntx_w, ppmi=False):
    Nw = target.get(tgt_w)
    Nc = context.get(cntx_w)
    Nwc = D.get((tgt_w, cntx_w))
    
    if Nwc is None:
        return None
    if Nwc == 0:
        if ppmi:
            return 0
        else:
            return -math.inf

    val = math.log10( (Nwc * D_size) / (Nw * Nc) )
    if ppmi:
        val = max(val, 0)

    return val


# Construct PPMI matrix
row = []
col = []
data = []
for pair in D.keys():
    row.append(pair[0])
    col.append(pair[1])
    data.append(pmi(*pair, ppmi=True))
ppmi_matrix = sparse.csr_matrix((data, (row, col)), shape=(vocab_size, vocab_size))

# sparisfy
ppmi_matrix.eliminate_zeros()


#----------------- Release memory --------------------#
del vocab
del corp
del wi
del D
del context
del target
del row
del col
del data

#----------------- Construct Embeddings -------------------#
# SVD on PPMI Matrix (memory hungry)
u, sigma, vt = svds(ppmi_matrix, k = EMBEDDING_DIM)

# Get embeddings with reduced dimension
#embeddings = u * sigma
embeddings = u  # Eigenvalue weighting (see Levy et al. 2015)

# Normalize embeddings to unit length
    # See example: np.array([[1,2], [3, 4] ,[5,6]]) / [2,1] (column-wise operation)
scalar_array = np.linalg.norm(embeddings, axis=1)
embeddings = (embeddings.T / scalar_array).T


#----------------- Save Embeddings Data --------------------#
np.save(f"svd_ppmi_embeddings_{EMBEDDING_DIM}dim.npy", embeddings)


# Signal process done
beep(s)
