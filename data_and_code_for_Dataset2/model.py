import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from Contrast import Contrast
from Metapath_Augmentation.mp_encoder import Mp_encoder
from Structural_Augmentation.st_encoder import St_encoder
import os
from diffusion import Diffusion
import numpy as np
import networkx as nx
import dgl




class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super(CrossAttentionFusion, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.inter_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads, 
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        self.dropout = nn.Dropout(0.4)
        
    def forward(self, view1_emb, view2_emb):
        view1_emb = view1_emb.unsqueeze(1)
        view2_emb = view2_emb.unsqueeze(1)
        
        inter_out1, _ = self.inter_attention(view1_emb, view2_emb, view2_emb)
        inter_out1 = self.norm1(view1_emb + self.dropout(inter_out1))
        inter_out1 = self.norm2(inter_out1 + self.dropout(self.ff(inter_out1)))
        
        inter_out2, _ = self.inter_attention(view2_emb, view1_emb, view1_emb)
        inter_out2 = self.norm1(view2_emb + self.dropout(inter_out2))
        inter_out2 = self.norm2(inter_out2 + self.dropout(self.ff(inter_out2)))
        
        inter_out1 = inter_out1.squeeze(1)
        inter_out2 = inter_out2.squeeze(1)
        
        fused_emb = torch.cat([inter_out1, inter_out2], dim=-1)
        return fused_emb




def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_torch(seed=1234)



class PRMHGRL(nn.Module):
    def __init__(self, args):
        super(PRMHGRL, self).__init__()
        self.args = args
        self.mp = Mp_encoder(self.args)
        self.st = St_encoder(self.args)
        self.contrast_l = Contrast(self.args)
        self.contrast_d = Contrast(self.args)

        
        self.lncRNA_linear = nn.Linear(args.embedding_dim_RNA, args.hgt_in_dim)
        self.disease_linear = nn.Linear(args.embedding_dim_disease, args.hgt_in_dim)
        self.miRNA_linear = nn.Linear(args.embedding_dim_RNA, args.hgt_in_dim)        
  

        self.Diffusion_lncRNA_v1 = Diffusion()
        self.Diffusion_lncRNA_v2 = Diffusion()
        
        self.Diffusion_disease_v1 = Diffusion()
        self.Diffusion_disease_v2 = Diffusion()        

        self.hidden_dim = args.hgt_out_dim

        
        self.lncRNA_fusion = CrossAttentionFusion(
            embed_dim=args.hgt_out_dim,
            num_heads=8
        )
        
        self.disease_fusion = CrossAttentionFusion(
            embed_dim=args.hgt_out_dim,
            num_heads=8
        )
        

        self.mlp = nn.Sequential(
            nn.Linear(2*self.args.hgt_out_dim, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2)
        )




    def forward(self, g_lm, g_dm, g_l, g_d, pos_l, pos_d,
                lncRNA_feature, disease_feature, miRNA_feature, pos_sample, neg_sample):
        lncRNA_feature = F.elu(self.lncRNA_linear(lncRNA_feature))
        disease_feature = F.elu(self.disease_linear(disease_feature))
        miRNA_feature = F.elu(self.miRNA_linear(miRNA_feature))

        l_mp, d_mp = self.mp(g_lm, g_dm, lncRNA_feature, disease_feature, miRNA_feature)
        l_st, d_st = self.st(g_l, g_d, lncRNA_feature, disease_feature, miRNA_feature)


        lncRNA_sim_mp = cos_sim(l_mp)
        disease_sim_mp = cos_sim(d_mp)
        lncRNA_sim_st = cos_sim(l_st)
        disease_sim_st = cos_sim(d_st)


        graph_lncRNA_mp=self.get_graph(lncRNA_sim_mp, 4).to('cuda')
        graph_lncRNA_st=self.get_graph(lncRNA_sim_st, 4).to('cuda')
        graph_disease_mp=self.get_graph(disease_sim_mp, 4).to('cuda')
        graph_disease_st=self.get_graph(disease_sim_st, 4).to('cuda')

        l_mp_v2=l_mp.clone()
        l_st_v2=l_st.clone()
        d_mp_v2=d_mp.clone()
        d_st_v2=d_st.clone()


        l_diffu_lncRNA_v1,_ = self.Diffusion_lncRNA_v1(graph_lncRNA_mp, l_mp_v2, l_st_v2)
        l_diffu_lncRNA_v2,_ = self.Diffusion_lncRNA_v2(graph_lncRNA_st, l_st_v2, l_mp_v2)

        l_diffu_disease_v1,_ = self.Diffusion_disease_v1(graph_disease_mp, d_mp_v2, d_st_v2)
        l_diffu_disease_v2,_ = self.Diffusion_disease_v2(graph_disease_st, d_st_v2, d_mp_v2)

        l_l = self.contrast_l(l_mp_v2, l_st_v2, pos_l)
        l_d = self.contrast_d(d_mp_v2, d_st_v2, pos_d)

        l_l_all=0.7*l_l+0.3*(l_diffu_lncRNA_v1+l_diffu_lncRNA_v2)
        l_d_all=0.7*l_d+0.3*(l_diffu_disease_v1+l_diffu_disease_v2)

        l_mp_v3=l_mp_v2.clone()
        l_st_v3=l_st_v2.clone()
        d_mp_v3=d_mp_v2.clone()
        d_st_v3=d_st_v2.clone()        
        
        l_emb = self.lncRNA_fusion(l_mp_v3, l_st_v3)

        d_emb = self.disease_fusion(d_mp_v3, d_st_v3)
        
        
        lncdi_emb_pos = torch.mul(l_emb[pos_sample[:, 0]], d_emb[pos_sample[:, 1]])
        
        lncdi_emb_neg=torch.mul(l_emb[neg_sample[:, 0]], d_emb[neg_sample[:, 1]])


        lncdi_emb = torch.cat((lncdi_emb_pos, lncdi_emb_neg), dim=0)
        output = self.mlp(lncdi_emb)

        self.l_emb = l_emb
        self.d_emb = d_emb

        return l_l_all + l_d_all, output



    def get_graph(self, matrix, k):
        matrix = matrix.cpu().detach().numpy()
        num = matrix.shape[0]
        knn_graph = np.zeros(matrix.shape)
        idx_sort = np.argsort(-(matrix - np.eye(num)), axis=1)
        for i in range(num):
            knn_graph[i, idx_sort[i, :k + 1]] = matrix[i, idx_sort[i, :k + 1]]
            knn_graph[idx_sort[i, :k + 1], i] = matrix[idx_sort[i, :k + 1], i]
        knn_graph = knn_graph + np.eye(num)
        graph_nx = nx.from_numpy_array(knn_graph)
        graph = dgl.from_networkx(graph_nx)
        return graph


def cos_sim(feature_matrix):
    feature_matrix_norm = F.normalize(feature_matrix, p=2, dim=1)
    feature_matrix_sim = torch.mm(feature_matrix_norm, feature_matrix_norm.T)
    return feature_matrix_sim


