# Construct Word Embeddings with SVD & PPMI Matrix from Raw Text

## Building Embeddings

Set the parameters in `main.py`:

```python
RAW_CORP = 'path/to/corpus'  # txt with tokens separated by spaces
WINDOW = 3                   # left & right context window size
EMBEDDING_DIM = 300          # the dimension of output embeddings (watch for memory limits)
```

Then run:

```bash
python3 main.py
```

This should generate two files: `svd_ppmi_embeddings_vocab.pkl` and `svd_ppmi_embeddings_{EMBEDDING_DIM}dim.npy`.

- `svd_ppmi_embeddings_vocab.pkl`
    - A dictionary with words as keys and integers as values. The integers correspond to the row indices in the 2d array in `svd_ppmi_embeddings_{EMBEDDING_DIM}dim.npy`.
- `svd_ppmi_embeddings_{EMBEDDING_DIM}dim.npy`
    - A 2d numpy array with each row vector (length equals `EMBEDDING_DIM`) corresponding to a word embedding.


## Usage

```python
# See svd_embeddings.py
from svd_embeddings import Embeddings
embed = Embeddings()

>>> embed.cossim("哥哥", "姊姊")
0.9718541428549548

>>> embed.getWordvec("醫生")
array([ 0.02296084, -0.08675744,  0.07824458,  0.0681636 , -0.02192351,
        0.08818709, -0.02195274,  0.17403084, -0.04071053, -0.18709724,
        0.04536741, -0.07284144,  0.09114984, -0.05165448,  0.00451687,
       -0.04058027,  0.09230312, -0.07219502,  0.01216258, -0.04172952,
       -0.11094631, -0.07241607,  0.01181941,  0.05238818, -0.17830793,
        0.21766906,  0.08224388, -0.03238169,  0.10863629, -0.02812842,
        0.20527716, -0.00130599, -0.18120455, -0.10064474, -0.07525918,
       -0.24233707, -0.1017248 ,  0.03523229, -0.23127462, -0.30223162,
       -0.0248824 ,  0.22797739,  0.04060027,  0.15641568,  0.17344962,
       -0.16877309,  0.03490095,  0.21471432,  0.19750746,  0.46938252])
```