import torch
import torch.nn as nn
import random
import numpy as np
from net import Linear
from utils import genDataLoader, genDataLoader_with_action, generate_new_dataset
from tqdm import tqdm
import time
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F
import copy
from sklearn.model_selection import StratifiedKFold
import os

class PG1:
    def __init__(self, rna_id, pro_id, feas, labels, action_dim, data_set):

        self.data_set = data_set
        self.sents = feas
        self.labels = labels
        self.rna_id = rna_id
        self.pro_id = pro_id
        random.seed(1)
        random.shuffle(self.sents)
        random.seed(1)
        random.shuffle(self.labels)
        random.seed(1)
        random.shuffle(self.rna_id)
        random.seed(1)
        random.shuffle(self.pro_id)

        self.action_dim = action_dim
        self.action_list = [a for a in range(self.action_dim)]


        self.stable_epoch = 10
        self.early_stop_epoch = 10
        self.best_epoch = 9999
        self.best_score = 0
        self.best_action = []
        self.action_count_array = np.zeros((len(self.labels),), dtype=int)
        self.saved_log_probs = []
        self.saved_rewards = []
        self.rewards_types = []
        self.new_labels_classifier = []
        self.new_labels_policy = []
        self.mean_f1 = 0.0
        self.is_stable = False

        self.gamma = 0.95
        self.pretrain_lr = 0.001
        self.policy_network_lr = 0.001
        self.classifier_network_lr = 0.001
        self.pretrain_epoch = 20
        self.classifier_epoch = 10
        self.classifier_weight_decay = 0.01
        self.policy_weight_decay = 0.01


        self.num_episodes = 60
        self.batch = 120
        self.eps = np.finfo(np.float32).eps.item()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("Using Cuda")


        self.policy_network = self.pretrain_policy_network()
        self.policy_network_opt = torch.optim.Adam(self.policy_network.parameters(), lr=self.policy_network_lr, weight_decay=self.policy_weight_decay)
        self.policy_loader = genDataLoader(self.sents, self.labels, shuffle=False, batch_size=self.batch)


    def caculate_max(self):

        new_labels = self.labels

        y_counts = pd.value_counts(new_labels)
        print("Right Labels: ")
        print(y_counts)

        print('\n## Calculate Source ##')
        loss_fn = nn.CrossEntropyLoss()
        i = 0
        f1_list = []
        auc_list = []
        auprc_list = []
        stratifiedKFolds = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
        for (trn_idx, val_idx) in stratifiedKFolds.split(self.sents, new_labels):

            i += 1
            train_sents, test_sents = np.array(self.sents, dtype=np.float32)[trn_idx], np.array(self.sents, dtype=np.float32)[val_idx]
            train_labels, test_labels = np.array(new_labels)[trn_idx], np.array(new_labels)[val_idx]

            classifier_network = Linear(self.action_dim)
            policy_network = classifier_network.to(self.device)
            parameters = filter(lambda p: p.requires_grad, policy_network.parameters())
            optimizer = torch.optim.Adam(parameters, lr=self.classifier_network_lr,
                                         weight_decay=self.classifier_weight_decay)


            train_loader = genDataLoader(train_sents, train_labels, shuffle=True, batch_size=self.batch)
            test_loader = genDataLoader(test_sents, test_labels, shuffle=True, batch_size=self.batch)

            classifier_network.train()
            for epoch in range(self.classifier_epoch):
                for x, y in train_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    y_pred = classifier_network(x)
                    optimizer.zero_grad()

                    loss = loss_fn(y_pred, y.long())
                    loss.backward()
                    optimizer.step()

                classifier_network.eval()
                pred_list = []
                prob_list = []
                y_list = []
                for x, y in test_loader:
                    x = x.to(self.device)
                    with torch.no_grad():
                        prob = policy_network(x)
                    pred = torch.max(prob, 1)[1].view(y.size()).cpu().data.numpy().tolist()
                    prob_1 = F.softmax(prob, dim=1)[:, 1].cpu().data.numpy().tolist()
                    y = y.numpy().tolist()
                    prob_list += prob_1
                    pred_list += pred
                    y_list += y

                f1 = f1_score(y_list, pred_list)
                auc = roc_auc_score(y_list, prob_list)
                auprc = average_precision_score(y_list, prob_list)

                f1_list.append(f1)
                auc_list.append(auc)
                auprc_list.append(auprc)


        f1 = np.mean(f1_list)
        auc = np.mean(auc_list)
        auprc = np.mean(auprc_list)
        print("[Classifier] 10-fold result: F1: %f, auc: %f, auprc: %f" % (f1, auc, auprc))
        return auc

    def pretrain_policy_network(self):
        print('policy_network预训练开始......')
        policy_network = Linear(self.action_dim)
        policy_network = policy_network.to(self.device)
        parameters = filter(lambda p: p.requires_grad, policy_network.parameters())
        optimizer = torch.optim.Adam(parameters, lr=self.pretrain_lr, weight_decay=self.policy_weight_decay)
        loss_fn = nn.CrossEntropyLoss()
        pretrain_loader = genDataLoader(self.sents, self.labels, shuffle=True, batch_size=self.batch)

        policy_network.train()

        for epoch in range(1,self.pretrain_epoch+1):
            for (x, y) in tqdm(pretrain_loader):
                x, y = x.to(self.device), y.to(self.device)
                y_pred = policy_network(x)
                optimizer.zero_grad()

                loss = loss_fn(y_pred, y.long())
                loss.backward()
                optimizer.step()

            policy_network.eval()
            pred_list = []
            y_list = []
            for (x, y) in tqdm(pretrain_loader):
                x = x.to(self.device)
                with torch.no_grad():
                    prob = policy_network(x)
                pred = torch.max(prob, 1)[1].view(y.size()).cpu().data.numpy().tolist()
                y = y.numpy().tolist()
                pred_list += pred
                y_list += y

            neg_pred_true_num = 0
            neg_all_num = 0
            for y, pred in zip(y_list, pred_list):
                if y == 0:
                    neg_all_num += 1
                    if y == pred:
                        neg_pred_true_num += 1

            neg_accuracy = neg_pred_true_num/neg_all_num
            accuracy = accuracy_score(y_list, pred_list)

            print("\n[PreTrain acc] epoch: %d, acc: %f, neg_acc: %f\n" % (epoch, accuracy, neg_accuracy))

            if neg_accuracy >= 0.80 and epoch >= 5:
                break
                
                
        return policy_network

    def chose_action(self, x):


        with torch.no_grad():
            probs = self.policy_network(x)
        probs = F.softmax(probs, dim=1)
        if self.is_stable:
            action = probs.argmax(dim=1, keepdim=True)
        else:

            action = torch.multinomial(probs, num_samples=1, replacement=False)


        return action.squeeze(1).cpu().numpy().tolist()


    def calculate_classifier_source(self, new_labels=None):

        if not new_labels:
            new_labels = self.new_labels_classifier

        y_counts = pd.value_counts(new_labels)
        print("New Labels: ")
        print(y_counts)

        print('\n## Calculate Source ##')
        loss_fn = nn.CrossEntropyLoss()
        i = 0
        f1_list = []
        auc_list = []
        auprc_list = []
        stratifiedKFolds = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
        for (trn_idx, val_idx) in stratifiedKFolds.split(self.sents, new_labels):

            i += 1
            train_sents, test_sents = np.array(self.sents, dtype=np.float32)[trn_idx], np.array(self.sents, dtype=np.float32)[val_idx]
            train_labels, test_labels = np.array(new_labels)[trn_idx], np.array(new_labels)[val_idx]

            classifier_network = Linear(self.action_dim)
            policy_network = classifier_network.to(self.device)
            parameters = filter(lambda p: p.requires_grad, policy_network.parameters())
            optimizer = torch.optim.Adam(parameters, lr=self.classifier_network_lr,
                                         weight_decay=self.classifier_weight_decay)

            train_loader = genDataLoader(train_sents, train_labels, shuffle=True, batch_size=self.batch)
            test_loader = genDataLoader(test_sents, test_labels, shuffle=True, batch_size=self.batch)

            classifier_network.train()
            for epoch in range(self.classifier_epoch):
                for x, y in train_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    y_pred = classifier_network(x)
                    optimizer.zero_grad()

                    loss = loss_fn(y_pred, y.long())
                    loss.backward()
                    optimizer.step()

                classifier_network.eval()
                pred_list = []
                prob_list = []
                y_list = []
                for x, y in test_loader:
                    x = x.to(self.device)
                    with torch.no_grad():
                        prob = policy_network(x)
                    pred = torch.max(prob, 1)[1].view(y.size()).cpu().data.numpy().tolist()
                    prob_1 = F.softmax(prob, dim=1)[:, 1].cpu().data.numpy().tolist()
                    y = y.numpy().tolist()
                    prob_list += prob_1
                    pred_list += pred
                    y_list += y


                if len(set(y_list)) <= 1:
                    return 0.5

                f1 = f1_score(y_list, pred_list)
                auc = roc_auc_score(y_list, prob_list)
                auprc = average_precision_score(y_list, prob_list)

                f1_list.append(f1)
                auc_list.append(auc)
                auprc_list.append(auprc)


        f1 = np.mean(f1_list)
        auc = np.mean(auc_list)
        auprc = np.mean(auprc_list)
        print("[Classifier] 10-fold result: F1: %f, auc: %f, auprc: %f" % (f1, auc, auprc))
        return auc

    def calculate_rewards(self, episode):
        f1 = self.calculate_classifier_source()
        f1_r = f1 - self.mean_f1
        print("\ncurrent_score: %f  mean_score: %f  score_r: %f" % (f1, self.mean_f1, f1_r))


        if f1 > self.mean_f1:
            self.mean_f1 = f1
        for t in self.rewards_types:
            if t == -1:
                if f1_r > 0:
                    self.saved_rewards.append(f1_r)
                else:
                    self.saved_rewards.append(f1_r/2)
            elif t == -2:
                if f1_r > 0:
                    self.saved_rewards.append(f1_r/2)
                else:
                    self.saved_rewards.append(f1_r)
            elif t == 0:
                self.saved_rewards.append(-abs(f1_r))
            elif t == 1:

                self.saved_rewards.append(0)
            else:
                assert False

        return f1

    def finish_episode(self):

        print("Updating Policy network...")
        self.policy_network.train()
        loader = genDataLoader_with_action(self.sents, self.labels, self.new_labels_policy, shuffle=False, batch_size=self.batch)
        for x, y, a in loader:
            x, a = x.to(self.device), a.to(self.device)

            probs = self.policy_network(x)
            probs = F.softmax(probs, dim=1)
            chose_probs = probs.gather(1, a)
            for prob in chose_probs:
                logp = prob.log()
                self.saved_log_probs.append(logp)

        G = 0
        policy_loss = []
        G_list = []
        for r in self.saved_rewards[::-1]:
            G = r
            G_list.insert(0, G)
        G_list = torch.FloatTensor(G_list)
        G_list = (G_list - G_list.mean()) / (G_list.std() + self.eps)
        for log_prob, G in zip(self.saved_log_probs, G_list):
            policy_loss.append(-log_prob * G)
        policy_loss = torch.cat(policy_loss).sum()
        self.policy_network_opt.zero_grad()
        policy_loss.backward()
        self.policy_network_opt.step()

        del self.saved_log_probs[:]
        del self.saved_rewards[:]
        del self.rewards_types[:]
        del self.new_labels_classifier[:]
        del self.new_labels_policy[:]

    def train_policy(self):
        self.new_labels_classifier = copy.deepcopy(self.labels)
        self.mean_f1 = self.calculate_classifier_source()
        del self.new_labels_classifier[:]

        for i in range(self.num_episodes):

            episode_time = time.time()

            print("\n*****************[RL] epoch: %d*********************" % (i))
            print("Selecting action")

            if i >= self.stable_epoch:
                self.is_stable = True

            self.policy_network.eval()
            for x, y in self.policy_loader:
                x = x.to(self.device)
                a = self.chose_action(x)
                y = y.numpy().tolist()
                for aa, yy in zip(a, y):
                    self.new_labels_policy.append(aa)
                    if yy == 1:
                        self.new_labels_classifier.append(1)
                        if aa == 1:
                            self.rewards_types.append(1)
                        elif aa == 0:
                            self.rewards_types.append(0)
                        else:
                            assert False
                    elif yy == 0:
                        self.new_labels_classifier.append(aa)
                        if aa == 1:
                            self.rewards_types.append(-2)
                        elif aa == 0:
                            self.rewards_types.append(-1)
                        else:
                            assert False
                    else:
                        assert False


            pos_idx = np.where(np.array(self.labels) == 1)[0]
            pos_acc = sum(np.array(self.new_labels_policy)[pos_idx]) / pos_idx.shape[0]
            print("pos_acc:", pos_acc)

            score = self.calculate_rewards(i)
            rAll = sum(self.saved_rewards)

            if self.is_stable:

                self.action_count_array += np.array(self.new_labels_policy)

                if score > self.best_score:
                    self.best_score = score
                    print("\nbest score: %f\n" % (self.best_score))
                    self.best_action = copy.deepcopy(self.new_labels_classifier)
                    self.best_epoch = i

                if i - self.best_epoch >= self.early_stop_epoch:
                    print("********Early stop*******")

                    neg_true_false_idx = np.array(self.labels) == 0
                    neg_count = sum(neg_true_false_idx)
                    print("original neg data num:", neg_count)


                    pos_rna = []
                    pos_pro = []


                    output_dir = "./result/" + self.data_set
                    os.makedirs(output_dir, exist_ok=True)

                    epoch_after_stable = i - self.stable_epoch + 1

                    limit_rate = 0.8
                    new_labels = []
                    correct_num = 0
                    limit_num = int(limit_rate * epoch_after_stable)
                    for is_n, action_count in zip(neg_true_false_idx, self.action_count_array):
                        if is_n:
                            if action_count >= limit_num:
                                new_labels.append(1)
                                correct_num += 1
                            else:
                                new_labels.append(0)
                        else:
                            new_labels.append(1)

                    print("limit rate: %f, correct num: %d" % (limit_rate, correct_num))


                    retain_rna = []

                    retain_pro = []

                    retain_features = []

                    for r, p, fea, ol, nl in zip(self.rna_id, self.pro_id, self.sents, self.labels, new_labels):
                        if ol == 0:
                            if nl == 0:

                                retain_rna.append(r)
                                retain_pro.append(p)
                                retain_features.append(fea)
                    
                    scores=[]
                    with torch.no_grad():
                        prob_temp=self.policy_network(torch.FloatTensor(np.array(retain_features)).to(self.device))
                        prob_1_temp = F.softmax(prob_temp, dim=1)[:,1].cpu().data.numpy().tolist()                   
                        scores.append(prob_1_temp)
                                                          
                                
                    output_file = os.path.join(output_dir, self.data_set)
                    generate_new_dataset(pos_rna, pos_pro, retain_rna, retain_pro, scores, output_file)
                    break

            self.finish_episode()

            print("Episode [%d/%d] reward: %f took: %.5fs \n" % (i, self.num_episodes, rAll, time.time() - episode_time))
