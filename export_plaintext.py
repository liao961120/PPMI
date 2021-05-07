#%%
import re
import tqdm
from  zhon import hanzi
from svd_embeddings import Embeddings

emb = Embeddings(embed_dim=50)
pat = re.compile('[{}]'.format(hanzi.characters))

outfile = []
for w in tqdm.tqdm(emb.wi):
    if len(w) == 0: continue
    if any( not pat.match(char) for char in w): 
        continue
    
    i = emb.wi[w]
    vec = '\t'.join(str(x) for x in emb.wordvec[i])
    outfile.append(f"{w}\t{vec}")

with open("svd_ppmi_embeddings_50dim.txt", "w") as f:
    f.write('\n'.join(outfile) + '\n')
