import timeit
import argparse
import numpy as np
import pandas as pd
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as fn
import os
from data_preprocess import *
from Metapath_Augmentation.path_aug_model import Metapath_Augmentation
from Structural_Augmentation.struc_aug import Structural_Augmentation
from pos_contrast import mp_pos, mp_data

from model import PRMHGRL
from metric import *
import warnings

warnings.filterwarnings('ignore')
device = torch.device('cuda')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=150, help='number of epochs to train')
    parser.add_argument('--k_fold', type=int, default=5, help='k-fold cross validation')    
    parser.add_argument('--lr', type=float, default=0.0004, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight_decay')
    parser.add_argument('--random_seed', type=int, default=1234, help='random seed')
    parser.add_argument('--negative_rate', type=float, default=1.0, help='negative_rate')
    parser.add_argument('--dataset', default='Dataset2', help='dataset')
    parser.add_argument('--dropout', default='0.15', type=float, help='dropout of HGT')
    parser.add_argument('--augmentation_path', default=['LDL', 'LML', 'DLD', 'DMD'], type=list, help='augmentation_path')
    parser.add_argument('--augmentation_inter_graph_num', default=1, type=int, help='augmentation_inter_graph_num')
    parser.add_argument('--graphon_method', default='USVT', help='method of graphon estimation')
    parser.add_argument('--threshold_usvt', default='0.1', type=float, help='threshold of usvt')
    parser.add_argument('--lam_l', default='0.4', type=float, help='coefficient of mixup')
    parser.add_argument('--lam_d', default='0.3', type=float, help='coefficient of mixup')
    parser.add_argument('--n', default='2', type=int, help='n power of the adjacency matrix')
    parser.add_argument('--p_lncRNA', default='30', type=int, help='threshold of lncRNA node degree')
    parser.add_argument('--p_disease', default='20', type=int, help='threshold of disease node degree')
 
    parser.add_argument('--t', default='0.8', type=float, help='threshold of feature similarity')
    parser.add_argument('--pos_num', default='5', type=int, help='threshold of positive samples selection')
    parser.add_argument('--hgt_layer', default='2', type=int, help='heterogeneous graph transformer layer')   
    parser.add_argument('--hgt_head', default='4', type=int, help='heterogeneous graph transformer head')
    parser.add_argument('--embedding_dim_disease', default='405', type=int, help='disease initial embedding dimension')
    parser.add_argument('--embedding_dim_RNA', default='768', type=int, help='RNA initial embedding dimension')    
    parser.add_argument('--hgt_in_dim', default='128', type=int, help='heterogeneous graph transformer input dimension')
    parser.add_argument('--hgt_head_dim', default='32', type=int, help='heterogeneous graph transformer head dimension')
    parser.add_argument('--hgt_out_dim', default='128', type=int, help='heterogeneous graph transformer output dimension')
    
    parser.add_argument('--tau_cross', default='0.6', type=float, help='tau in cross-view contrastive loss')
    parser.add_argument('--tau_mp', default='0.6', type=float, help='tau_mp in intra-view contrastive loss')
    parser.add_argument('--tau_sc', default='0.6', type=float, help='tau_sc in intra-view contrastive loss')
    parser.add_argument('--lam_cross', default='0.5', type=float, help='lam in cross-view contrastive loss')
    parser.add_argument('--lam_intra', default='0.5', type=float, help='lam in intra-view contrastive loss')
    parser.add_argument('--loss_rate', default='0.5', type=float, help='loss rate of unsupervised learning and training')

    parser.add_argument('--test_ratio', default='0.2', type=float, help='ratio of internal holdout test set')
    parser.add_argument('--model_dir', default='saved_models', help='directory to save models')

    args = parser.parse_args()
    args.data_dir = 'data/' + args.dataset + '/'
    
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    data = get_data(args)
    args.lncRNA_number = data['lncRNA_number']
    args.disease_number = data['disease_number']
    args.miRNA_number = data['miRNA_number']

    data = data_processing(data, args)
    
    data = split_independent_test(data, args)
    data = k_fold(data, args)


    lncRNA_feature = torch.FloatTensor(data['lncRNA_init_emb']).to(device)
    disease_feature = torch.FloatTensor(data['disease_init_emb']).to(device)
    miRNA_feature = torch.FloatTensor(data['miRNA_init_emb']).to(device)


    start = timeit.default_timer()

    cross_entropy = nn.CrossEntropyLoss()

    Metric = ('Epoch\t\tTime\t\tLoss_con\t\tLoss_cls\t\tAUC\t\tAUPR\t\tAccuracy\t\tPrecision\t\tRecall\t\tF1-score\t\tMcc')
    AUCs, AUPRs = [], []

    print('Dataset:', args.dataset)

    for i in range(args.k_fold):

        print('fold:', i)
        print(Metric)

        model = PRMHGRL(args)
        model = model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='max', factor=0.8, patience=100, verbose=True, min_lr=0.0004)

        best_auc, best_aupr, best_accuracy, best_precision, best_recall, best_f1, best_mcc = 0, 0, 0, 0, 0, 0, 0
        X_train_p = torch.LongTensor(data['X_train_p'][i]).to(device)
        X_train_n = torch.LongTensor(data['X_train_n'][i]).to(device)
        Y_train = torch.LongTensor(data['Y_train'][i]).to(device)
        X_test_p = torch.LongTensor(data['X_test_p'][i]).to(device)
        X_test_n = torch.LongTensor(data['X_test_n'][i]).to(device)
        Y_test = data['Y_test'][i].flatten()

        lncdimi_graph, meta_paths, edge_types, data = dgl_heterograph(data, data['X_train_p'][i], args)

        g_s_lncRNA, feature_graph_lncRNA = Structural_Augmentation(data, args, 'lncRNA', lncdimi_graph, edge_types)
        g_s_disease, feature_graph_disease = Structural_Augmentation(data, args, 'disease', lncdimi_graph, edge_types)
        g_m_lncRNA, g_m_disease, g_adj_l, g_adj_d = Metapath_Augmentation(lncdimi_graph, meta_paths, edge_types, args)

        ldl, dld = mp_data(data['X_train_p'][i], data, args)
        pos_l, pos_d = mp_pos(ldl, dld, feature_graph_lncRNA, feature_graph_disease, args)
        lncdimi_graph = lncdimi_graph.to(device)
        g_s_lncRNA = g_s_lncRNA.to(device)
        g_s_disease = g_s_disease.to(device)
        g_m_lncRNA = g_m_lncRNA.to(device)
        g_m_disease = g_m_disease.to(device)

        pos_l = torch.IntTensor(pos_l.toarray()).to(device)
        pos_d = torch.IntTensor(pos_d.toarray()).to(device)


        for epoch in range(args.epochs):

            model.train()

            l_con, train_output = model(g_m_lncRNA, g_m_disease, g_s_lncRNA, g_s_disease, pos_l, pos_d,
                                        lncRNA_feature, disease_feature, miRNA_feature, X_train_p, X_train_n)                                        
            l_train = cross_entropy(train_output, torch.flatten(Y_train))
            l_all = l_con + l_train
            optimizer.zero_grad()
            l_all.backward()
            optimizer.step()

            l_con = l_con.detach().cpu().numpy()
            l_train = l_train.detach().cpu().numpy()

            with torch.no_grad():
                model.eval()

                _, test_score = model(g_m_lncRNA, g_m_disease, g_s_lncRNA, g_s_disease,
                                      pos_l, pos_d, lncRNA_feature, disease_feature, miRNA_feature, X_test_p, X_test_n)                                      
            test_prob = fn.softmax(test_score, dim=-1)
            test_score = torch.argmax(test_score, dim=-1)

            test_prob = test_prob[:, 1]
            test_prob = test_prob.cpu().numpy()

            test_score = test_score.cpu().numpy()

            AUC, AUPR, accuracy, precision, recall, f1, mcc = get_metric(Y_test, test_score, test_prob)

            scheduler.step(AUC)

            end = timeit.default_timer()
            time = end - start
            show = [epoch + 1, round(time, 2), l_con, l_train, round(AUC, 5), round(AUPR, 5), round(accuracy, 5),
                       round(precision, 5), round(recall, 5), round(f1, 5), round(mcc, 5)]
            print('\t\t'.join(map(str, show)))
            if AUC > best_auc:
                best_epoch = epoch + 1
                best_auc = AUC
                best_aupr, best_accuracy, best_precision, best_recall, best_f1, best_mcc = AUPR, accuracy, precision, recall, f1, mcc
                print('AUC improved at epoch ', best_epoch, ';\tbest_auc:', best_auc)
                
                model_path = os.path.join(args.model_dir, f'best_model_fold_{i}.pt')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_auc': best_auc,
                    'best_aupr': best_aupr,
                    'args': args
                }, model_path)
                print(f'Model saved to {model_path}')

                np.savetxt(str(i+1)+'true_valid.txt',Y_test,fmt='%f',delimiter=' ')
                np.savetxt(str(i+1)+'predict_valid.txt',test_prob,fmt='%f',delimiter=' ')

        AUCs.append(best_auc)
        AUPRs.append(best_aupr)

    print('Cross-validation AUC:', AUCs)
    AUC_mean = np.mean(AUCs)
    AUC_std = np.std(AUCs)
    print('Mean CV AUC:', AUC_mean, '(', AUC_std, ')')

    print('Cross-validation AUPR:', AUPRs)
    AUPR_mean = np.mean(AUPRs)
    AUPR_std = np.std(AUPRs)
    print('Mean CV AUPR:', AUPR_mean, '(', AUPR_std, ')')

    print('\nEvaluating on internal holdout test set...')

    X_ind_test_p = torch.LongTensor(data['X_ind_test_p']).to(device)
    X_ind_test_n = torch.LongTensor(data['X_ind_test_n']).to(device)
    Y_ind_test = data['Y_ind_test'].flatten()

    ind_AUCs, ind_AUPRs = [], []
    ind_accuracies, ind_precisions, ind_recalls, ind_f1s, ind_mccs = [], [], [], [], []

    for i in range(args.k_fold):
        print(f'Evaluating fold {i+1} model on internal holdout test set...')
        
        model = PRMHGRL(args)
        model = model.to(device)
        
        model_path = os.path.join(args.model_dir, f'best_model_fold_{i}.pt')
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Loaded model from {model_path} (epoch {checkpoint["epoch"]+1}, AUC: {checkpoint["best_auc"]:.5f})')
        
        lncdimi_graph, meta_paths, edge_types, data = dgl_heterograph(data, data['X_train_p'][i], args)
        
        g_s_lncRNA, feature_graph_lncRNA = Structural_Augmentation(data, args, 'lncRNA', lncdimi_graph, edge_types)
        g_s_disease, feature_graph_disease = Structural_Augmentation(data, args, 'disease', lncdimi_graph, edge_types)
        g_m_lncRNA, g_m_disease, g_adj_l, g_adj_d = Metapath_Augmentation(lncdimi_graph, meta_paths, edge_types, args)
        
        ldl, dld = mp_data(data['X_train_p'][i], data, args)
        pos_l, pos_d = mp_pos(ldl, dld, feature_graph_lncRNA, feature_graph_disease, args)
        lncdimi_graph = lncdimi_graph.to(device)
        g_s_lncRNA = g_s_lncRNA.to(device)
        g_s_disease = g_s_disease.to(device)
        g_m_lncRNA = g_m_lncRNA.to(device)
        g_m_disease = g_m_disease.to(device)
        
        pos_l = torch.IntTensor(pos_l.toarray()).to(device)
        pos_d = torch.IntTensor(pos_d.toarray()).to(device)
        
        
        with torch.no_grad():
            model.eval()
            _, ind_test_score = model(g_m_lncRNA, g_m_disease, g_s_lncRNA, g_s_disease,
                                      pos_l, pos_d, lncRNA_feature, disease_feature, miRNA_feature, 
                                      X_ind_test_p, X_ind_test_n)        
        ind_test_prob = fn.softmax(ind_test_score, dim=-1)
        ind_test_score = torch.argmax(ind_test_score, dim=-1)
        
        ind_test_prob = ind_test_prob[:, 1]
        ind_test_prob = ind_test_prob.cpu().numpy()
        
        ind_test_score = ind_test_score.cpu().numpy()

        np.savetxt(str(i+1)+'true_inter_holdout_test.txt',Y_ind_test,fmt='%f',delimiter=' ')
        np.savetxt(str(i+1)+'predict_inter_holdout_test.txt',ind_test_prob,fmt='%f',delimiter=' ')
        
        fold_AUC, fold_AUPR, fold_accuracy, fold_precision, fold_recall, fold_f1, fold_mcc = get_metric(Y_ind_test, ind_test_score, ind_test_prob)
        
        ind_AUCs.append(fold_AUC)
        ind_AUPRs.append(fold_AUPR)
        ind_accuracies.append(fold_accuracy)
        ind_precisions.append(fold_precision)
        ind_recalls.append(fold_recall)
        ind_f1s.append(fold_f1)
        ind_mccs.append(fold_mcc)
        
        print(f'Fold {i+1} on Internal Holdout Test: AUC: {fold_AUC:.5f}, AUPR: {fold_AUPR:.5f}, Accuracy: {fold_accuracy:.5f}')
        print(f'Precision: {fold_precision:.5f}, Recall: {fold_recall:.5f}, F1-score: {fold_f1:.5f}, MCC: {fold_mcc:.5f}')

    ind_AUC_mean = np.mean(ind_AUCs)
    ind_AUC_std = np.std(ind_AUCs)
    ind_AUPR_mean = np.mean(ind_AUPRs)
    ind_AUPR_std = np.std(ind_AUPRs)
    ind_accuracy_mean = np.mean(ind_accuracies)
    ind_precision_mean = np.mean(ind_precisions)
    ind_recall_mean = np.mean(ind_recalls)
    ind_f1_mean = np.mean(ind_f1s)
    ind_mcc_mean = np.mean(ind_mccs)

    print('\nInternal Holdout Test Results (Average over all folds):')
    print(f'AUC: {ind_AUC_mean:.5f} (±{ind_AUC_std:.5f})')
    print(f'AUPR: {ind_AUPR_mean:.5f} (±{ind_AUPR_std:.5f})')
    print(f'Accuracy: {ind_accuracy_mean:.5f}')
    print(f'Precision: {ind_precision_mean:.5f}, Recall: {ind_recall_mean:.5f}')
    print(f'F1-score: {ind_f1_mean:.5f}, MCC: {ind_mcc_mean:.5f}')



