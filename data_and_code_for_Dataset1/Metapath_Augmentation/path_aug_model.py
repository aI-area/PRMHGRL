import numpy as np
import dgl
from Metapath_Augmentation import learner
from Metapath_Augmentation import simulator
from scipy import sparse
import itertools




class Path_Augmentation:
    def __init__(self, g, meta_path_dic, method, args):
        super().__init__()

        self.meta_path_dic = meta_path_dic
        self.g = g
        self.meta_path_reach = {}
        self.method = method
        self.args = args
        for meta_path in meta_path_dic:
            self.meta_path_reach[meta_path] = dgl.metapath_reachable_graph(g, meta_path_dic[meta_path])


    def estimate_graphon(self, meta_path):

        mat = self.meta_path_reach[meta_path].adj(scipy_fmt='coo').toarray()
        
        graphon = learner.estimate_graphon([mat], self.method, self.args)
        return graphon


    def generate_graph(self, node_num, num_graphs, graphon):
        num_nodes = node_num
        generated_graphs = simulator.simulate_graphs(graphon, num_graphs=num_graphs, num_nodes=num_nodes, graph_size="fixed")      
        for ind, item in enumerate(generated_graphs):
            generated_graphs[ind] = item + np.eye(node_num)
            
        return generated_graphs


def metapath_augmentation(g, path_augmentation, metapath_type, args, target_category, edge_types):

    graphons = {}
    node_num = g.ndata['h'][target_category].size()[0]
    for path in args.augmentation_path:
        graphons[path] = path_augmentation.estimate_graphon(path)
        

    augmentated_graphs = []

    arg_W = []
    combinations = list(itertools.combinations(metapath_type, 2))
    for com1, com2 in combinations:

        if target_category == 'lncRNA':
            new_graphon = args.lam_l * graphons[com1] + (1 - args.lam_l) * graphons[com2]
        else:
            new_graphon = args.lam_d * graphons[com1] + (1 - args.lam_d) * graphons[com2]
        arg_W.append(new_graphon)
        augmentated_graphs = augmentated_graphs + path_augmentation.generate_graph(node_num,
                                                                                   args.augmentation_inter_graph_num,
                                                                                   new_graphon)
    node_dict = {
        'disease': args.disease_number,
        'lncRNA': args.lncRNA_number,
        'miRNA': args.miRNA_number
    }


    hetero_dic = {}
    for edge in edge_types:
        hetero_dic[(edge_types[edge][0], edge, edge_types[edge][1])] = g[edge].edges()


    for ind, item in enumerate(augmentated_graphs):
        adj_mat = sparse.coo_matrix(item)
        
        hetero_dic[(target_category, "AUG_" + str(ind), target_category)] = (adj_mat.row, adj_mat.col)


    new_g = dgl.heterograph(hetero_dic,num_nodes_dict=node_dict)
    new_g.nodes[target_category].data['h'] = g.ndata['h'][target_category]

    return new_g, augmentated_graphs


def Metapath_Augmentation(lncdimi_graph, meta_paths, edge_types, args):
    path_augmentation = Path_Augmentation(lncdimi_graph, meta_paths, args.graphon_method, args)
    target_category = 'lncRNA'
    metapath_type = ['LDL', 'LML']
    g_m_lncRNA, g_adj_l = metapath_augmentation(lncdimi_graph, path_augmentation, metapath_type, args, target_category, edge_types)
    target_category = 'disease'
    metapath_type = ['DLD', 'DMD']
    g_m_disease, g_adj_d = metapath_augmentation(lncdimi_graph, path_augmentation, metapath_type, args, target_category, edge_types)
    return g_m_lncRNA, g_m_disease, g_adj_l, g_adj_d
