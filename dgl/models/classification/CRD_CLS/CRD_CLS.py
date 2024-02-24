import os
from dgl.nn import GraphConv
import torch.nn as nn
import torch.nn.functional as F

os.environ["DGLBACKEND"] = "pytorch"

class CRD_CLS(nn.Module):
    def __init__(self, num_features, hidden, num_classes, dropout=0.2):
        super(CRD_CLS, self).__init__()
        self.conv1 = GraphConv(num_features, hidden)
        self.p = dropout
        self.conv2 = GraphConv(hidden, num_classes)

    def forward(self, g, in_feat):
        x = self.conv1(g, in_feat)
        x = F.relu(x)
        x = F.dropout(x, p=self.p)
        x = self.conv2(g, x)
        x = F.log_softmax(x)
        return x
