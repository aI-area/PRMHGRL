import numpy as np
import scipy.sparse as sp


def mp_data(X_train_p, data, args):

    l = args.lncRNA_number
    d = args.disease_number
    m = args.miRNA_number
    lncmi = data['lncmi']
    dimi = data['dimi']

    label_l = np.array([1] * (len(X_train_p) + len(lncmi)), dtype=int)
    lncmi_mi = lncmi[:, -1] + d
    lncmi = np.array([lncmi[:, 0], lncmi_mi]).T
    lnc_dimi = np.concatenate((X_train_p, lncmi), axis=0)
    lnc_dimi = np.concatenate((lnc_dimi, np.expand_dims(label_l, axis=1)), axis=1)
    lnc_dimi = sp.coo_matrix((np.ones(lnc_dimi.shape[0]),(lnc_dimi[:,0], lnc_dimi[:, 1])), shape=(l, d + m)).toarray()
    ldl = np.matmul(lnc_dimi, lnc_dimi.T)
    ldl = sp.coo_matrix(ldl)

    label_d = np.array([1] * (len(X_train_p) + len(dimi)), dtype=int)
    dimi_mi = dimi[:, -1] + l
    dimi = np.array([dimi[:, 0], dimi_mi]).T
    di_lncmi = np.concatenate((X_train_p[:, (1, 0)], dimi), axis=0)
    di_lncmi = np.concatenate((di_lncmi, np.expand_dims(label_d, axis=1)), axis=1)
    di_lncmi = sp.coo_matrix((np.ones(di_lncmi.shape[0]), (di_lncmi[:, 0], di_lncmi[:, 1])), shape=(d, l + m)).toarray()
    dld = np.matmul(di_lncmi, di_lncmi.T)
    dld = sp.coo_matrix(dld)
    return ldl, dld


def mp_pos(ldl, dld, feature_graph_lncRNA, feature_graph_disease, args):
    l = args.lncRNA_number
    d = args.disease_number
    m = args.miRNA_number
    pos_num = args.pos_num
    ldl = ldl.A.astype("float32")
    dia_l = sp.dia_matrix((np.ones(l), 0), shape=(l, l)).toarray()

    ll = np.ones((l, l)) - dia_l
    ldl = ldl * ll
    pos_l_mp = np.zeros((l, l))
    k = 0
    for i in range(l):
        pos_l_mp[i, i] = 1
        one = ldl[i].nonzero()[0]
        if len(one) > pos_num - 1:
            oo = np.argsort(-ldl[i, one])
            sele = one[oo[:pos_num - 1]]
            pos_l_mp[i, sele] = 1
            k += 1
        else:
            pos_l_mp[i, one] = 1
    pos_l = pos_l_mp * feature_graph_lncRNA.A
    pos_l = sp.coo_matrix(pos_l)

    dld = dld.A.astype("float32")
    dia_d = sp.dia_matrix((np.ones(d), 0), shape=(d, d)).toarray()
    dd = np.ones((d, d)) - dia_d
    dld = dld * dd
    pos_d_mp = np.zeros((d, d))
    k = 0
    for j in range(d):
        pos_d_mp[j, j] = 1
        one = dld[j].nonzero()[0]
        if len(one) > pos_num - 1:
            oo = np.argsort(-dld[j, one])
            sele = one[oo[:pos_num - 1]]
            pos_d_mp[j, sele] = 1
            k += 1
        else:
            pos_d_mp[j, one] = 1
    pos_d = pos_d_mp * feature_graph_disease.A
    pos_d = sp.coo_matrix(pos_d)
    return pos_l, pos_d

