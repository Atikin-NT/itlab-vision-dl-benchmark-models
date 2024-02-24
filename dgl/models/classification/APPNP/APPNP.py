import os
from dgl.nn import GraphConv
import torch.nn as nn
import torch.nn.functional as F

os.environ["DGLBACKEND"] = "pytorch"

class APPNP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.2, layer=2):
        super(APPNP, self).__init__()
        self.gc1 = GraphConv(nfeat, nhid) # n_feature: C, n_hidden: H
        self.gc_layer = GraphConv(nhid, nhid)  # n_hidden: H, n_hidden: H
        self.gc2 = GraphConv(nhid, nclass) # n_hidden: H, n_classes: F
        self.dropout = dropout
        self.layer = layer

    def forward(self, g, in_feat): # X, A
        x = self.gc1(g, in_feat)
        x = F.relu(x) # for APPNP paper
        for i in range(self.layer - 2):
            x = self.gc_layer(g, x)
            x = F.relu(x)  # middle conv
            x = F.dropout(x, self.dropout)
        if self.layer > 1:
            x = self.gc2(g, x) # 2th conv
        return F.log_softmax(x) # N * F
