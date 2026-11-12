import torch
import torch.nn as nn
import dgl
import dgl.nn.pytorch
import torch.nn.functional as F


class HGT(nn.Module):
    def __init__(self, args):
        super(HGT, self).__init__()
        self.layer = args.hgt_layer
        self.head = args.hgt_head
        self.in_dim = args.hgt_in_dim      
        self.hgt = nn.ModuleList()
        for _ in range(args.hgt_layer):
            self.hgt.append(dgl.nn.pytorch.conv.HGTConv(args.hgt_in_dim, int(args.hgt_in_dim / args.hgt_head), args.hgt_head, 3, 7, args.dropout))
            
        self.lncRNA_linear = nn.Linear(args.hgt_in_dim, args.hgt_in_dim)
        self.disease_linear = nn.Linear(args.hgt_in_dim, args.hgt_in_dim)
        self.miRNA_linear = nn.Linear(args.hgt_in_dim, args.hgt_in_dim)

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(args.hgt_in_dim) for _ in range(args.hgt_layer)
        ])

        self.activation = nn.GELU()
        self.output_dropout = nn.Dropout(args.dropout)
        self.output_layer_norm = nn.LayerNorm(args.hgt_in_dim)


    def forward(self, hetero_g, lncRNA_feature, disease_feature, miRNA_feature):
    
        hetero_g.nodes['disease'].data['h'] = disease_feature
        hetero_g.nodes['lncRNA'].data['h'] = lncRNA_feature
        hetero_g.nodes['miRNA'].data['h'] = miRNA_feature

        homo_g = dgl.to_homogeneous(hetero_g, ndata='h')

        node_features = homo_g.ndata['h']
        node_types = homo_g.ndata['_TYPE']


        for i, (hgt_layer, layer_norm) in enumerate(zip(self.hgt, self.layer_norms)):
            node_features = hgt_layer(homo_g, node_features, node_types, homo_g.edata['_TYPE'], presorted=True)
            node_features = layer_norm(node_features)
            node_features = self.activation(node_features)
            
        
        node_features = self.output_layer_norm(node_features)
        node_features = self.output_dropout(node_features)

        disease_mask = (node_types == 0)
        lncRNA_mask = (node_types == 1)
        miRNA_mask = (node_types == 2)      

        lncRNA_out = node_features[lncRNA_mask]
        disease_out = node_features[disease_mask]
        miRNA_out = node_features[miRNA_mask]
        
        return lncRNA_out,disease_out,miRNA_out


class Mp_encoder(nn.Module):
    def __init__(self, args):
        super(Mp_encoder, self).__init__()
        
        self.lncRNA_hgt = HGT(args)
        self.disease_hgt = HGT(args)

    def forward(self, g_l, g_d, lncRNA_feature, disease_feature, miRNA_feature):
        emb_l,_,_ = self.lncRNA_hgt(g_l, lncRNA_feature, disease_feature, miRNA_feature)
        _,emb_d,_= self.disease_hgt(g_d, lncRNA_feature, disease_feature, miRNA_feature)

        return emb_l, emb_d
