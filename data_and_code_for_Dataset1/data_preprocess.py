import numpy as np
import random
import torch
import pandas as pd
import dgl
import networkx as nx
from sklearn.model_selection import StratifiedKFold
import copy
import os

device = torch.device('cuda')
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_fixed_independent_samples(data, args):
    pos_path = os.path.join(MODULE_DIR, 'inter_holdout_positive.txt')
    neg_path = os.path.join(MODULE_DIR, 'inter_holdout_negative.txt')
    if not os.path.exists(pos_path) or not os.path.exists(neg_path):
        raise FileNotFoundError(
            'Fixed independent-test files are missing: inter_holdout_positive.txt and inter_holdout_negative.txt'
        )

    ind_pos = np.atleast_2d(np.loadtxt(pos_path, dtype=int))
    ind_neg = np.atleast_2d(np.loadtxt(neg_path, dtype=int))
    if ind_pos.shape[1] != 2 or ind_neg.shape[1] != 2:
        raise ValueError('Each independent-test file must contain exactly two columns.')

    for name, samples in [('inter_holdout_positive.txt', ind_pos), ('inter_holdout_negative.txt', ind_neg)]:
        if np.any(samples[:, 0] < 0) or np.any(samples[:, 0] >= args.lncRNA_number):
            raise ValueError(f'{name} contains an invalid lncRNA index.')
        if np.any(samples[:, 1] < 0) or np.any(samples[:, 1] >= args.disease_number):
            raise ValueError(f'{name} contains an invalid disease index.')
        if len({tuple(row) for row in samples}) != len(samples):
            raise ValueError(f'{name} contains duplicate association pairs.')

    known_pairs = {tuple(row) for row in data['lncdi']}
    pos_pairs = {tuple(row) for row in ind_pos}
    neg_pairs = {tuple(row) for row in ind_neg}
    if not pos_pairs.issubset(known_pairs):
        raise ValueError('inter_holdout_positive.txt contains pairs that are not known associations.')
    if neg_pairs.intersection(known_pairs):
        raise ValueError('inter_holdout_negative.txt contains known positive associations.')
    if pos_pairs.intersection(neg_pairs):
        raise ValueError('The positive and negative independent-test sets overlap.')

    np.savetxt(pos_path, ind_pos, fmt='%d', delimiter=' ')
    np.savetxt(neg_path, ind_neg, fmt='%d', delimiter=' ')
    return ind_pos, ind_neg


def _exclude_pairs(samples, excluded_pairs, sample_name):
    excluded = {tuple(row) for row in excluded_pairs}
    available = {tuple(row[:2]) for row in samples}
    missing = excluded.difference(available)
    if missing:
        raise ValueError(f'{sample_name} is missing {len(missing)} fixed independent-test pairs.')
    keep_mask = np.array([tuple(row[:2]) not in excluded for row in samples], dtype=bool)
    return samples[keep_mask]


def _save_fold_files(fold_number, X_train_p, X_train_n, X_valid_p, X_valid_n,
                     Y_train, Y_valid):
    sample_train = np.vstack((X_train_p, X_train_n)).astype(int)
    sample_valid = np.vstack((X_valid_p, X_valid_n)).astype(int)
    np.savetxt(os.path.join(MODULE_DIR, f'sample{fold_number}train.txt'),
               sample_train, fmt='%d', delimiter=' ')
    np.savetxt(os.path.join(MODULE_DIR, f'sample{fold_number}valid.txt'),
               sample_valid, fmt='%d', delimiter=' ')
    np.savetxt(os.path.join(MODULE_DIR, f'label{fold_number}train.txt'),
               Y_train.astype(int), fmt='%d')
    np.savetxt(os.path.join(MODULE_DIR, f'label{fold_number}valid.txt'),
               Y_valid.astype(int), fmt='%d')

def get_adj(edges, size):
    edges_tensor = torch.LongTensor(edges).t()
    values = torch.ones(len(edges))
    adj = torch.sparse.LongTensor(edges_tensor, values, size).to_dense().long()
    adj = adj.numpy()
    return adj



def get_data(args):
    data = dict()

    lncRNA_init_emb = np.loadtxt(args.data_dir + 'lncRNA_embedding_LLM.txt', dtype=np.float64)
    disease_init_emb = np.loadtxt(args.data_dir + 'DSS.txt', dtype=np.float64)
    miRNA_init_emb = np.loadtxt(args.data_dir + 'miRNA_embedding_LLM.txt', dtype=np.float64)

    data['lncRNA_number'] = int(lncRNA_init_emb.shape[0])
    data['disease_number'] = int(disease_init_emb.shape[0])
    data['miRNA_number'] = int(miRNA_init_emb.shape[0])

    data['lncdi'] = pd.read_csv(args.data_dir + 'LncRNADiseaseAssociation.csv', dtype=int).to_numpy()
    data['lncmi'] = pd.read_csv(args.data_dir + 'LncRNAMiRNAAssociation.csv', dtype=int).to_numpy()
    data['dimi'] = pd.read_csv(args.data_dir + 'DiseaseMiRNAAssociation.csv', dtype=int).to_numpy()    

    data['dilnc'] = data['lncdi'][:, [1, 0]]
    data['milnc'] = data['lncmi'][:, [1, 0]]
    data['midi'] = data['dimi'][:, [1, 0]]
    
    data['lncRNA_init_emb'] = lncRNA_init_emb
    data['disease_init_emb'] = disease_init_emb
    data['miRNA_init_emb'] = miRNA_init_emb
    return data


def data_processing(data, args):
    lncdi_matrix = get_adj(data['lncdi'], (args.lncRNA_number, args.disease_number))
    one_index = []
    zero_index = []
    for i in range(lncdi_matrix.shape[0]):
        for j in range(lncdi_matrix.shape[1]):
            if lncdi_matrix[i][j] >= 1:
                one_index.append([i, j])

    negative_samples=np.loadtxt(args.data_dir + 'final_negative.txt')
    for i in range(negative_samples.shape[0]):
        temp1=negative_samples[i][0]
        temp2=negative_samples[i][1]
        zero_index.append([temp1, temp2])

    random.seed(args.random_seed)
    random.shuffle(one_index)
    random.shuffle(zero_index)

    zero_index = zero_index[:int(args.negative_rate * len(one_index))]

    index = np.array(one_index + zero_index, dtype=int)
    label = np.array([1] * len(one_index) + [0] * len(zero_index), dtype=int)
    samples = np.concatenate((index, np.expand_dims(label, axis=1)), axis=1)
    label_p = np.array([1] * len(one_index), dtype=int)

    lncdi_p = samples[samples[:, 2] == 1, :]
    lncdi_n = samples[samples[:, 2] == 0, :]

    
    data['all_samples'] = samples
    data['all_lncdi'] = samples[:, :2]
    data['all_lncdi_p'] = lncdi_p
    data['all_lncdi_n'] = lncdi_n
    data['all_label'] = label
    data['all_label_p'] = label_p
    return data


def k_fold(data, args):
    k_fold = args.k_fold
    all_pos = data['all_lncdi_p'][:, :2]
    all_neg = data['all_lncdi_n'][:, :2]
    
    min_samples = min(len(all_pos), len(all_neg))
    all_pos = all_pos[:min_samples]
    all_neg = all_neg[:min_samples]
    
    pos_indices = list(range(min_samples))
    neg_indices = list(range(min_samples))
    random.shuffle(pos_indices)
    random.shuffle(neg_indices)
    
    fold_size = min_samples // k_fold
    
    data['X_train_p'] = []
    data['X_train_n'] = []
    data['X_test_p'] = []
    data['X_test_n'] = []
    data['Y_train'] = []
    data['Y_test'] = []
    
    for i in range(k_fold):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i != k_fold - 1 else min_samples
        
        test_pos_idx = pos_indices[test_start:test_end]
        test_neg_idx = neg_indices[test_start:test_end]
        
        train_pos_idx = list(set(pos_indices) - set(test_pos_idx))
        train_neg_idx = list(set(neg_indices) - set(test_neg_idx))
        
        X_train_p = all_pos[train_pos_idx]
        X_train_n = all_neg[train_neg_idx]
        X_test_p = all_pos[test_pos_idx]
        X_test_n = all_neg[test_neg_idx]
        
        Y_train = np.concatenate([np.ones(len(train_pos_idx)), np.zeros(len(train_neg_idx))])
        Y_test = np.concatenate([np.ones(len(test_pos_idx)), np.zeros(len(test_neg_idx))])

        _save_fold_files(i + 1, X_train_p, X_train_n, X_test_p, X_test_n,
                         Y_train, Y_test)
        
        train_shuffle_idx = list(range(len(Y_train)))
        test_shuffle_idx = list(range(len(Y_test)))
        random.shuffle(train_shuffle_idx)
        random.shuffle(test_shuffle_idx)
        
        data['X_train_p'].append(X_train_p)
        data['X_train_n'].append(X_train_n)
        data['X_test_p'].append(X_test_p)
        data['X_test_n'].append(X_test_n)
        data['Y_train'].append(Y_train)
        data['Y_test'].append(Y_test)
        
        print(f"Fold {i+1}:")
        print(f"Training: X_train_p: {len(X_train_p)}, X_train_n: {len(X_train_n)}")
        print(f"Testing: X_test_p: {len(X_test_p)}, X_test_n: {len(X_test_n)}")
    
    return data



def dgl_heterograph(data, lncdi, args):
    lncdi_list, lncmi_list, dimi_list = [], [], []
    dilnc_list, milnc_list, midi_list = [], [], []
    dilnc = lncdi[:, [1, 0]]

    for i in range(lncdi.shape[0]):
        lncdi_list.append(lncdi[i])
        dilnc_list.append(dilnc[i])
    for i in range(data['lncmi'].shape[0]):
        lncmi_list.append(data['lncmi'][i])
        milnc_list.append(data['milnc'][i])
    for i in range(data['dimi'].shape[0]):
        dimi_list.append(data['dimi'][i])
        midi_list.append(data['midi'][i])

    node_dict = {
        'disease': args.disease_number,
        'lncRNA': args.lncRNA_number,
        'miRNA': args.miRNA_number
    }


    heterograph_dict = {
        ('lncRNA', 'lncRNA-disease', 'disease'): (lncdi_list),
        ('disease', 'disease-lncRNA', 'lncRNA'): (dilnc_list),
        ('lncRNA', 'lncRNA-miRNA', 'miRNA'): (lncmi_list),
        ('miRNA', 'miRNA-lncRNA', 'lncRNA'): (milnc_list),
        ('disease', 'disease-miRNA', 'miRNA'): (dimi_list),
        ('miRNA', 'miRNA-disease', 'disease'): (midi_list)
    }

    data['feature_dict'] = {
        'lncRNA': torch.tensor(data['lncRNA_init_emb']),
        'disease': torch.tensor(data['disease_init_emb']),
        'miRNA': torch.tensor(data['miRNA_init_emb'])
    }

    lncdimi_graph = dgl.heterograph(heterograph_dict, num_nodes_dict=node_dict)
    lncdimi_graph.ndata['h'] = data['feature_dict']

    meta_paths = {'LDL':['lncRNA-disease', 'disease-lncRNA'],
                  'LML':['lncRNA-miRNA', 'miRNA-lncRNA'],
                  'DLD':['disease-lncRNA', 'lncRNA-disease'],
                  'DMD':['disease-miRNA', 'miRNA-disease']}

    edge_types = {
        'lncRNA-disease': ['lncRNA', 'disease'],
        'disease-lncRNA': ['disease', 'lncRNA'],
        'lncRNA-miRNA': ['lncRNA', 'miRNA'],
        'miRNA-lncRNA': ['miRNA', 'lncRNA'],
        'disease-miRNA': ['disease', 'miRNA'],
        'miRNA-disease': ['miRNA', 'disease']
    }

    data['lncdimi_graph'] = lncdimi_graph
    data['meta_paths'] = meta_paths
    data['edge_types'] = edge_types

    return lncdimi_graph, meta_paths, edge_types, data


def split_independent_test(data, args):

    ind_test_pos, ind_test_neg = _load_fixed_independent_samples(data, args)
    pos_samples = data['all_lncdi_p']
    neg_samples = data['all_lncdi_n']
    cv_pos = _exclude_pairs(pos_samples, ind_test_pos, 'positive sample pool')
    cv_neg = _exclude_pairs(neg_samples, ind_test_neg, 'negative sample pool')

    data['X_ind_test_p'] = ind_test_pos
    data['X_ind_test_n'] = ind_test_neg
    pos_labels = np.ones((len(ind_test_pos), 1))
    neg_labels = np.zeros((len(ind_test_neg), 1))
    data['Y_ind_test'] = np.vstack((pos_labels, neg_labels))
    data['all_lncdi_p'] = cv_pos
    data['all_lncdi_n'] = cv_neg

    print(f"Independent test set: {len(ind_test_pos)} positive samples, {len(ind_test_neg)} negative samples")
    print(f"Cross-validation set: {len(cv_pos)} positive samples, {len(cv_neg)} negative samples")
    return data


