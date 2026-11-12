import os
import numpy as np

def read_features_files(pairs_file):
    rna_stru_fea = np.loadtxt('lncRNA_embedding_LLM.txt',dtype=np.float64)
    pro_stru_fea = np.loadtxt('DSS.txt',dtype=np.float64)

    rna_id = []
    pro_id = []

    rna_id2 = []
    pro_id2 = []


    f2 = open(pairs_file, 'r')
    for line2 in f2.readlines():
        line2 = line2.strip()
        rna_id2.append(line2.split()[0])
        pro_id2.append(line2.split()[1])
    f2.close()


    with open(pairs_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            rna_id.append(int(parts[0]))
            pro_id.append(int(parts[1]))

    features = []

    for r, p in zip(rna_id, pro_id):
        combined = np.concatenate([rna_stru_fea[r], pro_stru_fea[p]])
        features.append(combined)  
    
    return features, rna_id2, pro_id2


