import sys
sys.path.insert(0, '/Users/princess/Documents/PhD/thesis/code/2aurum-datadiscovery-master')

debug=1
argumentPassing=1

import sys
from scipy.sparse import csc_matrix
import numpy as np
#from main import init_system
from collections import namedtuple
#from modelstore.elasticstore import KWType
from api.apiutils import Relation
# from algebra import API, DRS
#from mock import MagicMock, patch
#from ddapi import API
#from modelstore.elasticstore import StoreHandler
from knowledgerepr.fieldnetwork import deserialize_network
from knowledgerepr.fieldnetwork import serialize_network_to_csv
from knowledgerepr.fieldnetwork import serialize_network_to_json
from knowledgerepr.fieldnetwork import serialize_network_to_onecsv
from knowledgerepr.fieldnetwork import serialize_network_to_onecsv_detailed

from collections import defaultdict
from collections import OrderedDict
import itertools
from DoD.utils import FilterType
from inputoutput import inputoutput as io

from modelstore.elasticstore import StoreHandler

# Machine Learning
import dgl
# import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy as sc
from scipy import stats
from scipy.sparse import csr_matrix
import scipy.io
import urllib.request
import dgl.function as fn

#calculate the percentage of elements smaller than the k-th element
def perc(input,k): return sum([1 if i else 0 for i in input<input[k]])/float(len(input))

class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        super(HeteroRGCNLayer, self).__init__()
        # W_r for each relation
        self.weight = nn.ModuleDict({
                name : nn.Linear(in_size, out_size) for name in etypes
            })

    def forward(self, G, feat_dict):
        # The input is a dictionary of node features for each type
        funcs = {}
        for srctype, etype, dsttype in G.canonical_etypes:
            # Compute W_r * h
            Wh = self.weight[etype](feat_dict[srctype])
            # Save it in graph for message passing
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            # Specify per-relation message passing functions: (message_func, reduce_func).
            # Note that the results are saved to the same destination feature 'h', which
            # hints the type wise reducer for aggregation.
            funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))
        # Trigger message passing of multiple types.
        # The first argument is the message passing functions for each relation.
        # The second one is the type wise reducer, could be "sum", "max",
        # "min", "mean", "stack"
        G.multi_update_all(funcs, 'sum')
        # return the updated node feature dictionary

        return {ntype : G.nodes[ntype].data['h'] for ntype in G.ntypes}

###############################################################################
# Create a simple GNN by stacking two ``HeteroRGCNLayer``. Since the
# nodes do not have input features, make their embeddings trainable.

class HeteroRGCN(nn.Module):
    def __init__(self, G, in_size, hidden_size, out_size):
        super(HeteroRGCN, self).__init__()
        # Use trainable node embeddings as featureless inputs.
        embed_dict = {ntype : nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), in_size))
                      for ntype in G.ntypes}
        for key, embed in embed_dict.items():
            nn.init.xavier_uniform_(embed)
        self.embed = nn.ParameterDict(embed_dict)
        # create layers
        self.layer1 = HeteroRGCNLayer(in_size, hidden_size, G.etypes)
        self.layer2 = HeteroRGCNLayer(hidden_size, out_size, G.etypes)

        self.dropout = nn.Dropout(p=0.3)


    def forward(self, G):
        h_dict = self.layer1(G, self.embed)
        h_dict = {k : F.leaky_relu(h) for k, h in h_dict.items()}
        h_dict = self.layer2(G, h_dict)
        # get paper logits
        # return (h_dict['attribute'], h_dict['table'])
        return h_dict['attribute']


class activelearning():
    def __init__(self):
        self.amap = {}
        self.amap_inv = {}
        self.tmap = {}
        self.relations  = []
        self.relationmap = {}
        self.data = {}
        self.train_labels = []
        self.train_nids = []
        self.valid_labels = []
        self.valid_samples = []
        self.labels = []
        self.selection = 'random'
        self.learning_curve = {}
        # self.nids_true = []
        # self.nids_false = []
        # self.nids_true_idx = np.array([])
        # self.nids_false_idx = np.array([])

    def preprocessgraph(self, network):
    
        # =======================
        # Prepare matrix representation for each relation type

        table_to_ids = network._get_underlying_repr_table_to_ids()
        G = network._get_underlying_repr_graph()
        nodes = list(G.nodes)
        edges = list(G.edges)
        ntables = len(table_to_ids)

        # create node map
        titer = 0
        aiter = 0
        self.amap = {}
        self.amap_inv = {}
        self.tmap = {}
        for n in nodes:
            try:
                if int(n) >= 10 and int(n) <= 10+ntables:
                # if r == Relation.CONTAINER:
                    if n not in self.tmap:
                        self.tmap[n] = titer
                        titer += 1
                else:
                    if n not in self.amap:
                        self.amap[n] = aiter
                        self.amap_inv[aiter] = n
                        aiter += 1
            except:
                continue

       
        if debug == 1:
            print(ntables)
            print()
            print (self.amap)
            print()
            print (self.tmap)
            print()


        # Fill the matrices
        self.relations  = []
        self.relationmap = { }
        for (n1,n2,r) in edges:       
            if r not in self.relations:
                self.relations.append(r)
            
            if r not in self.relationmap:
                self.relationmap[r] = {}
                self.relationmap[r]['rows'] = np.array([],dtype=np.int64)
                self.relationmap[r]['cols'] = np.array([],dtype=np.int64)
                self.relationmap[r]['data'] = np.array([],dtype=np.int64)

            if r == Relation.CONTAINER:
                self.relationmap[r]['rows'] = np.append(self.relationmap[r]['rows'], [self.amap[n1]])
                self.relationmap[r]['cols'] = np.append(self.relationmap[r]['cols'], [self.tmap[n2]])
                self.relationmap[r]['data'] = np.append(self.relationmap[r]['data'], [1])
            else:
                self.relationmap[r]['rows'] = np.append(self.relationmap[r]['rows'], [self.amap[n1]])
                self.relationmap[r]['cols'] = np.append(self.relationmap[r]['cols'], [self.amap[n2]])
                self.relationmap[r]['data'] = np.append(self.relationmap[r]['data'], [1])

            

        # create the adjancency matrices for each relation
        self.ntables = len(self.tmap)
        self.nattributes = len(self.amap)
        for r in self.relations:

            if r == Relation.CONTAINER:
                self.relationmap[r]['adjmat'] = csc_matrix((self.relationmap[r]['data'], (self.relationmap[r]['rows'], self.relationmap[r]['cols'])), shape=(self.nattributes, self.ntables))  
            else:
                self.relationmap[r]['adjmat'] = csc_matrix((self.relationmap[r]['data'], (self.relationmap[r]['rows'], self.relationmap[r]['cols'])), shape=(self.nattributes, self.nattributes))  
            if debug == 1:
                print(self.relations)
                print(self.relationmap)
                print()
                print(self.relationmap[r]['adjmat'])
                print()
                print(self.relationmap[r]['adjmat'].toarray())
                print()

        if debug == 1:
            print("\n\n-----------------\n")
            for r in self.relations:

                print(r)
                print(self.relationmap[r]['adjmat'].getnnz())
                print(self.relationmap[r]['adjmat'].shape[0])
                print(self.relationmap[r]['adjmat'].shape[1])
                print()
        return self.relationmap

    def preprocess_indices(self, k_samples_nids, train_labels):
        
        self.train_idx = np.array([self.amap.get(key) for key in k_samples_nids])
        self.val_idx = np.array([self.amap.get(key) for key in self.amap.keys()]) #list(self.amap.keys()))
        self.val_idx = np.delete(self.val_idx, self.train_idx, axis=0)

        if debug == 1:
            print()
            print("preprocess_indices() - train_idx:")
            print(self.train_idx)
            print()
            print("preprocess_indices() - val_idx:")
            print(self.val_idx)
            print()
               
        self.train_idx = torch.tensor(self.train_idx).long()
        self.val_idx = torch.tensor(self.val_idx).long()

        train_labels = np.array(train_labels)
        self.labels = np.zeros(len(self.amap))
        self.labels[self.train_idx] = train_labels
        self.labels = torch.tensor(self.labels).long()

       
    def mainlogic(self, whichmodel, k_samples_nids , labels):
        
        # whichmodel = '4'

        if '1' in whichmodel:
            path = '/Users/princess/Documents/PhD/thesis/code/2aurum-datadiscovery-master/mytest/testmodel1/'
        elif '2' in whichmodel:
            path = '/Users/princess/Documents/PhD/thesis/code/2aurum-datadiscovery-master/mytest/testmodel2/'
        elif '3' in whichmodel:
            path = '/Users/princess/Documents/PhD/thesis/code/2aurum-datadiscovery-master/mytest/testmodel3/'
        elif '4' in whichmodel:
            path = '/Users/princess/Documents/PhD/thesis/code/2aurum-datadiscovery-master/mytest/testmodel4/'
        

        #=======================
        # initialization
        
        store = StoreHandler()
        self.network = deserialize_network(path)


        # preprocess the graph
        self.data = self.preprocessgraph(self.network)
        
        # preprocess indexes
        self.preprocess_indices(k_samples_nids,labels)
        # labels = torch.tensor(labels).long()

        # prepare the graph for learning
        # create a reverse relationship as well

        self.G = dgl.heterograph({
            ('attribute', 'containby', 'table') : self.data[Relation.CONTAINER]['adjmat'],
            ('table', 'contain', 'attribute') : self.data[Relation.CONTAINER]['adjmat'].transpose(),
            ('attribute', 'Content_sim', 'attribute') : self.data[Relation.CONTENT_SIM]['adjmat'],
            ('attribute', 'Content_sim', 'attribute') : self.data[Relation.CONTENT_SIM]['adjmat'].transpose(),
            ('attribute', 'schema_sim', 'attribute') : self.data[Relation.SCHEMA_SIM]['adjmat'],
            ('attribute', 'schema_sim', 'attribute') : self.data[Relation.SCHEMA_SIM]['adjmat'].transpose(),
            ('attribute', 'pkfk', 'attribute') : self.data[Relation.PKFK]['adjmat'],
            ('attribute', 'pkfk', 'attribute') : self.data[Relation.PKFK]['adjmat'].transpose(),})

        if debug == 1:
            print(self.G)

        # ==============
        # start learning

        # prepare the training labels
        # k_samples_nids = np.array(['485805923', '2905051006','3909970351','3232918382','446531856','1670272379'])
        
        # k_samples = [self.amap.get(key) for key in k_samples_nids]
        
        # self.labels = 

        # build the model
        self.model = HeteroRGCN(self.G, 10, 10, 2)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)

        # train
        self.logits = self.model(self.G)

        if debug == 1:
            print ( )
            print ("mainlogic() - after training:")
            print(self.G.ntypes)
            print([self.G.number_of_nodes(ntype) for ntype in self.G.ntypes])
            print ( )
            print ("mainlogic() - logits:")  
            print(self.logits) 
            print("mainlogic() - logits[train_idx]")             
            print(self.logits[self.train_idx])
            print("mainlogic() - labels[train_idx]")
            print(self.labels[self.train_idx])
            print("\nmainlogic() - size():")
            print(self.logits[self.train_idx].size())
            print(self.labels[self.train_idx].size())


        # The loss is computed only for labeled nodes.
        self.loss = F.cross_entropy(self.logits[self.train_idx], self.labels[self.train_idx])

        self.pred = self.logits.argmax(1)
        if debug == 1:
            print()
            print("Predictions:")
            print(self.pred)

        self.train_acc = (self.pred[self.train_idx] == self.labels[self.train_idx]).float().mean()
        # val_acc = (pred[val_idx] == labels[val_idx]).float().mean()
        # test_acc = (pred[test_idx] == labels[test_idx]).float().mean()

        # if best_val_acc < val_acc:
        #     best_val_acc = val_acc
        #     best_test_acc = test_acc

        self.opt.zero_grad()
        self.loss.backward()
        self.opt.step()

        if debug == 1 or debug == 2:
            # print ( )
            # print ("First iteration of learning")
            print('Loss %.4f, Train Acc %.4f' % (
                    self.loss.item(),
                    self.train_acc.item(),
                    
                ))



    def ask_oracle(self, num_samples):
        
        if self.selection == 'random':
            select = np.random.choice(self.val_idx,num_samples,replace=False)
            select_idx = torch.tensor(select).long()
            if debug == 1:
                print()
                print ('ask_oracle() - select') 
                print (select)
                print ('ask_oracle() - select_idx')
                print (select_idx)
                print()

        elif self.selection == 'entropy':
            # 2- Uncertainty Selection
            #choose instance to label based on entropy
            entropy = sc.stats.entropy(self.logits[self.val_idx].detach().T) 
            if debug == 1:
                print()
                print("entropy:")
                print(entropy)
            entrperc = np.asarray([perc(entropy,i) for i in range(len(entropy))])
            if debug == 1:
                print()
                print("entrperc:")
                print(entrperc)

            # 3- Get the selected from the validation set
            # select_list = []
            # for i in range(num_samples):
            #     np.argmax(entrperc)
            select = np.argpartition(entrperc, -num_samples)[-num_samples:]
            # select=np.argmax(entrperc) # To DO: make sure entropy values can represent node index??
            select_idx = torch.tensor([self.val_idx[select]]).long()
            # select = np.array(select)

            if debug == 1:
                print()
                print("selected nodes (select):")
                print(select)

                print()
                print("selected nodes (val_idx):")
                print(self.val_idx[select])
                print(self.labels[self.val_idx[select]])

                print()
                print("train_idx Before:")
                print(self.train_idx)

        # self.train_idx = torch.cat((self.train_idx,select_idx),0)  #k_samples.append(select) 
        # # val_idx = np.array([])
        # # val_idx = np.copy(train_idx)
        # self.val_idx = np.delete(self.val_idx, select_idx, axis=0)
        # self.val_idx = torch.tensor(self.val_idx).long()

        if debug == 1:
            print()
            print("ask_oracle() - amap_inv")
            print(self.amap_inv)
            for i in select_idx:
                print (i)

        # Finally get node ids that need to be labeled by the oracle        
        selected_nids = [  self.amap_inv[idx] for idx in select ]
        return selected_nids


    def oneiter(self, k_samples_nids , labels):
        
        max_queried = len(k_samples_nids)
        if max_queried not in self.learning_curve:
            self.learning_curve[max_queried] = {}
            self.learning_curve[max_queried]['training'] = 0
            self.learning_curve[max_queried]['validation'] = 0
            self.learning_curve[max_queried]['testing'] = 0
            self.learning_curve[max_queried]['loss'] = 0

        # 3- preprocess indexes
        self.preprocess_indices(k_samples_nids,labels)

        # 4- Train again
        # train
        # train
        self.logits = self.model(self.G)

        if debug == 1:
            print ( )
            print ("oneiter() - after training:")
            print(self.G.ntypes)
            print([self.G.number_of_nodes(ntype) for ntype in self.G.ntypes])
            print ( )
            print ("oneiter() - logits:")  
            print(self.logits) 
            print("oneiter() - of train_idx")             
            print(self.logits[self.train_idx])
            print(self.labels[self.train_idx])

        # The loss is computed only for labeled nodes.
        self.loss = F.cross_entropy(self.logits[self.train_idx], self.labels[self.train_idx])

        self.pred = self.logits.argmax(1)
        if debug == 1:
            print()
            print("Predictions:")
            print(self.pred)

        self.train_acc = (self.pred[self.train_idx] == self.labels[self.train_idx]).float().mean()
        # val_acc = (pred[val_idx] == labels[val_idx]).float().mean()
        # test_acc = (pred[test_idx] == labels[test_idx]).float().mean()

        # if best_val_acc < val_acc:
        #     best_val_acc = val_acc
        #     best_test_acc = test_acc

        self.opt.zero_grad()
        self.loss.backward()
        self.opt.step()

        if debug == 1 or debug == 2:
            # print ( )
            # print ("First iteration of learning")
            print('Loss %.4f, Train Acc %.4f' % (
                    self.loss.item(),
                    self.train_acc.item(),
                    
                ))

    def get_relevant_nodes(self):
        # relv = self.pred[self.train_idx] == self.labels[self.train_idx]
        if debug == 1:
            print()
            print('get_relevant_nodes(): Predictions')
            print(self.pred)
            print((self.pred == 1).nonzero())
            print()

        relv_indices = (self.pred == 1).nonzero()
        if debug == 1:
            print()
            print("get_relevant_nodes(): Relevant indices:")
            for i in relv_indices.tolist():
                print(i[0],)
            # print(relv_indices)
            print()
        relv_nids = [  self.amap_inv[idx[0]] for idx in relv_indices.tolist() ]
        if debug == 1:
            print()
            print("get_relevant_nodes(): Relevant nids:")
            print(relv_nids)
            print()
        return relv_nids

    def get_relevant_nidsANDtables(self):
        relv_indices = (self.pred == 1).nonzero()
        relv_nids = [  self.amap_inv[idx[0]] for idx in relv_indices.tolist() ]

    def get_prediction_nodes(self, selected_nids):
        # relv = self.pred[self.train_idx] == self.labels[self.train_idx]
        if debug == 0:
            print()
            print('get_prediction_nodes(): Predictions')
            print(self.pred)
            print((self.pred == 1).nonzero())
            print()

        relv_indices = (self.pred == 1).nonzero()
        if debug == 0:
            print()
            print("get_relevant_nodes(): Relevant indices:")
            for i in relv_indices.tolist():
                print(i[0],)
            # print(relv_indices)
            print()
        relv_nids = [  self.amap_inv[idx[0]] for idx in relv_indices.tolist() ]
        if debug == 0:
            print()
            print("get_relevant_nodes(): Relevant nids:")
            print(relv_nids)
            print()
        return relv_nids

def preprocessgraph(whichmodel):
    
    if '1' in whichmodel:
        path = '/Users/princess/Documents/PhD/thesis/code/2aurum-datadiscovery-master/mytest/testmodel1/'
    elif '2' in whichmodel:
        path = '/Users/princess/Documents/PhD/thesis/code/2aurum-datadiscovery-master/mytest/testmodel2/'
    elif '3' in whichmodel:
        path = '/Users/princess/Documents/PhD/thesis/code/2aurum-datadiscovery-master/mytest/testmodel3/'
    elif '4' in whichmodel:
        path = '/Users/princess/Documents/PhD/thesis/code/2aurum-datadiscovery-master/mytest/testmodel4/'
    
    #=======================
    # initialization
    
    store = StoreHandler()

    # ddapi:
    network = deserialize_network(path)
    
    # =======================
    # Prepare matrix representation for each relation type

    # id_to_info = network._get_underlying_repr_id_to_field_info()
    table_to_ids = network._get_underlying_repr_table_to_ids()
    # print(len(table_to_ids))
    # print(table_to_ids)
    # print()
    # tablescnt = network.get_number_tables()
    # gcnt = network.graph_order()
    G = network._get_underlying_repr_graph()
    nodes = list(G.nodes)
    edges = list(G.edges)
    adjmat = network._get_underlying_repr_adjmat()
    ntables = len(table_to_ids)

    # create node map
    titer = 0
    aiter = 0
    amap = {}
    tmap = {}
    for n in nodes:
        try:
            if int(n) >= 10 and int(n) <= 10+ntables:
            # if r == Relation.CONTAINER:
                if n not in tmap:
                    tmap[n] = titer
                    titer += 1
            else:
                if n not in amap:
                    amap[n] = aiter
                    aiter += 1
        except:
            continue

   
    if debug == 1:
        print(ntables)
        print()
        print (amap)
        print()
        print (tmap)
        print()


    # Fill the matrices
    relations  = []
    relationmap = { }
    for (n1,n2,r) in edges:       
        if r not in relations:
            relations.append(r)
        
        if r not in relationmap:
            relationmap[r] = {}
            relationmap[r]['rows'] = np.array([],dtype=np.int64)
            relationmap[r]['cols'] = np.array([],dtype=np.int64)
            relationmap[r]['data'] = np.array([],dtype=np.int64)

        if r == Relation.CONTAINER:
            relationmap[r]['rows'] = np.append(relationmap[r]['rows'], [amap[n1]])
            relationmap[r]['cols'] = np.append(relationmap[r]['cols'], [tmap[n2]])
            relationmap[r]['data'] = np.append(relationmap[r]['data'], [1])
        else:
            relationmap[r]['rows'] = np.append(relationmap[r]['rows'], [amap[n1]])
            relationmap[r]['cols'] = np.append(relationmap[r]['cols'], [amap[n2]])
            relationmap[r]['data'] = np.append(relationmap[r]['data'], [1])

        

    # create the adjancency matrices for each relation
    ntables = len(tmap)
    nattributes = len(amap)
    for r in relations:

        if r == Relation.CONTAINER:
            relationmap[r]['adjmat'] = csc_matrix((relationmap[r]['data'], (relationmap[r]['rows'], relationmap[r]['cols'])), shape=(nattributes, ntables))  
        else:
            relationmap[r]['adjmat'] = csc_matrix((relationmap[r]['data'], (relationmap[r]['rows'], relationmap[r]['cols'])), shape=(nattributes, nattributes))  
        if debug == 1:
            print(relations)
            print(relationmap)
            print()
            print(relationmap[r]['adjmat'])
            print()
            print(relationmap[r]['adjmat'].toarray())
            print()

    if debug == 1:
        print("\n\n-----------------\n")
        for r in relations:

            print(r)
            print(relationmap[r]['adjmat'].getnnz())
            print(relationmap[r]['adjmat'].shape[0])
            print(relationmap[r]['adjmat'].shape[1])
            print()
    return relationmap

k = 4 # initial sample size
max_queried_list = [20] #[4,10,20, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
selection = 'entropy'


def main():
	
    whichmodel = '4'

    if '1' in whichmodel:
        path = '/Users/princess/Documents/PhD/thesis/code/2aurum-datadiscovery-master/mytest/testmodel1/'
    elif '2' in whichmodel:
        path = '/Users/princess/Documents/PhD/thesis/code/2aurum-datadiscovery-master/mytest/testmodel2/'
    elif '3' in whichmodel:
        path = '/Users/princess/Documents/PhD/thesis/code/2aurum-datadiscovery-master/mytest/testmodel3/'
    elif '4' in whichmodel:
        path = '/Users/princess/Documents/PhD/thesis/code/2aurum-datadiscovery-master/mytest/testmodel4/'
    


    k_samples_nids = np.array(['485805923', '2905051006','3909970351','3232918382','446531856','1670272379'])
    labels = np.array([1,1,1,1,1,1])
    modelobj = activelearning()
    modelobj.mainlogic(path, k_samples_nids,labels )
    selected_nids = modelobj.ask_oracle()
    print(selected_nids)
    k_samples_nids = np.append(k_samples_nids, selected_nids)
    labels = np.append(labels, [0])
    if debug == 1:
        print()
        print("main() - k_samlpe_nids: after")    
        print(k_samples_nids)
        print(labels)

    modelobj.oneiter(k_samples_nids, labels)



    return
    #=======================
    # initialization
    
    store = StoreHandler()

    # ddapi:
    network = deserialize_network(path)




    # preprocess the graph
    data = preprocessgraph('4')

    # prepare the graph for learning
    # create a reverse relationship as well

    G = dgl.heterograph({
        ('attribute', 'containby', 'table') : data[Relation.CONTAINER]['adjmat'],
        ('table', 'contain', 'attribute') : data[Relation.CONTAINER]['adjmat'].transpose(),
        ('attribute', 'Content_sim', 'attribute') : data[Relation.CONTENT_SIM]['adjmat'],
        ('attribute', 'Content_sim', 'attribute') : data[Relation.CONTENT_SIM]['adjmat'].transpose(),
        ('attribute', 'schema_sim', 'attribute') : data[Relation.SCHEMA_SIM]['adjmat'],
        ('attribute', 'schema_sim', 'attribute') : data[Relation.SCHEMA_SIM]['adjmat'].transpose(),
        ('attribute', 'pkfk', 'attribute') : data[Relation.PKFK]['adjmat'],
        ('attribute', 'pkfk', 'attribute') : data[Relation.PKFK]['adjmat'].transpose(),})

    if debug == 1:
        print(G)

    # get the initial labeled samples

    # ==============
    # start learning

    # prepare the training labels
    k_samples_nids = np.array(['485805923', '2905051006','3909970351','3232918382','446531856','1670272379'])
    
    k_samples = [myDictionary.get(key) for key in keys]

    k_samples = np.array([0,4,5])
    
    # labels = 

    # build the model
    model = HeteroRGCN(G, 10, 10, 2)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    # train
    logits = model(G)

    if debug == 1:
        print ( )
        print ("after training:")
        print(G.ntypes)
        print([G.number_of_nodes(ntype) for ntype in G.ntypes])
        print ( )
        print ("logits:")  
        print(logits)              
        # print(logits[k_samples])
        # print(labels[k_samples])

    return 


if __name__== "__main__":

    main()
  

# python run_dod.py --model_path 'mytest/testmodel1/' --list_attributes '' --list_values 'purdue;'

