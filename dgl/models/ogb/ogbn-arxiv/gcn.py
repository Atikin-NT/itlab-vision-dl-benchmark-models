import dgl.nn.pytorch as dglnn
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class GCN(nn.Module):
    def __init__(
        self,
        in_feats: int,
        n_hidden: int,
        n_classes: int,
        n_layers: int,
        activation,  # function
        dropout: float,
        use_linear: bool,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.use_linear = use_linear

        self.convs = nn.ModuleList()
        if use_linear:  # для вариации с линейным слоем
            self.linear = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(n_layers):
            in_hidden = n_hidden if i > 0 else in_feats
            out_hidden = n_hidden if i < n_layers - 1 else n_classes
            bias = i == n_layers - 1

            self.convs.append(
                dglnn.GraphConv(in_hidden, out_hidden, "both", bias=bias)
            )
            if use_linear:  # для вариации с линейным слоем
                self.linear.append(nn.Linear(in_hidden, out_hidden, bias=False))
            if i < n_layers - 1:
                self.norms.append(nn.BatchNorm1d(out_hidden))

        self.input_drop = nn.Dropout(min(0.1, dropout))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, graph, feat):
        h = feat
        # h = self.input_drop(h)

        for i in range(self.n_layers):
            conv = self.convs[i](graph, h)

            if self.use_linear:
                linear = self.linear[i](h)
                h = conv + linear
            else:
                h = conv

            # if i < self.n_layers - 1:
            #     h = self.norms[i](h)
            #     h = self.activation(h)
            #     h = self.dropout(h)

        return h


    def inference(self, graph, features, dataloader, storage_device):
        buffer_device = torch.device("cpu")
        x = features.read("node", None, "feat")

        # Применяем input_drop только один раз в начале
        # x = self.input_drop(x)

        for layer_idx, layer in enumerate(self.convs):
            is_last_layer = layer_idx == len(self.convs) - 1

            y = torch.empty(
                (graph.total_num_nodes,
                self.n_classes if is_last_layer else self.n_hidden),
                dtype=torch.float32,
                device=buffer_device
            )

            for data in tqdm(dataloader):
                block = data.blocks[0]

                hidden_x = layer(block, data.node_features["feat"])
                # if not is_last_layer:
                #     hidden_x = self.norms[layer_idx](hidden_x)
                #     hidden_x = self.activation(hidden_x)
                #     hidden_x = self.dropout(hidden_x)
                y[data.seeds[0] : data.seeds[-1] + 1] = hidden_x
                # By design, our output nodes are contiguous.
            if not is_last_layer:
                features.update("node", None, "feat", y)
            x = y

        return y

model = GCN(in_feats, 256, n_classes, 3, F.relu, 0.75, False)
