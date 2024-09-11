import numpy as np
from tensorflow import keras
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2

from spektral.data.loaders import SingleLoader
from spektral.datasets.citation import Citation
from spektral.layers import GCNConv
from spektral.transforms import LayerPreprocess


class SGCN:
    def __init__(self, K):
        self.K = K

    def __call__(self, graph):
        out = graph.a
        for _ in range(self.K - 1):
            out = out.dot(out)
        out.sort_indices()
        graph.a = out
        return graph


K = 2  # Propagation steps for SGCN
data = "cora"
dataset = Citation(data, transforms=[LayerPreprocess(GCNConv), SGCN(K)])
mask_tr, mask_va, mask_te = dataset.mask_tr, dataset.mask_va, dataset.mask_te
loader_te = SingleLoader(dataset, sample_weights=mask_te)


l2_reg = 5e-6  # L2 regularization rate
learning_rate = 0.2  # Learning rate
epochs = 100  # Number of training epochs
patience = 3600  # Patience for early stopping
a_dtype = dataset[0].a.dtype  # Only needed for TF 2.1


N = dataset.n_nodes  # Number of nodes in the graph
F = dataset.n_node_features  # Original size of node features
n_out = dataset.n_labels  # Number of classes


@keras.saving.register_keras_serializable()
class SimpleGCNNet(Model):
    def __init__(self):
        super().__init__()
        self.gcnconv = GCNConv(n_out, activation="softmax", kernel_regularizer=l2(l2_reg), use_bias=False)

    def call(self, inputs):
        x, a = inputs
        out = self.gcnconv([x, a])

        return out