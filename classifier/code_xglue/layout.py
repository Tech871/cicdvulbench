import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class PredictionLayout(nn.Module):
    def __init__(
            self,
            num_classes=2,
            input_size=None,
            hidden_size=256,
            hidden_dropout_probability=0
    ):
        super().__init__()
        # self.dense = nn.Linear(args.hidden_size * 2, args.hidden_size)
        if input_size is None:
            input_size = hidden_size

        self.dense = nn.Linear(input_size, hidden_size)
        self.dropout = hidden_dropout_probability and nn.Dropout(hidden_dropout_probability)
        self.out_proj = nn.Linear(hidden_size, num_classes)

    def forward(self, inputs):
        x = self.dropout(inputs) if self.dropout else inputs
        x = self.dense(x.float())
        x = torch.tanh(x)
        x = self.dropout(x) if self.dropout else x
        x = self.out_proj(x)
        return x


class ReGGNN(nn.Module):
    """Gated GNN with residual connection"""

    def __init__(
            self,
            feature_dim_size,
            hidden_size,
            num_gnn_layers,
            dropout_probability,
            act=nn.functional.relu,
            residual=True,
            att_op='mul',
            alpha_weight=1.0
    ):
        super(ReGGNN, self).__init__()
        self.num_gnn_layers = num_gnn_layers
        self.residual = residual
        self.att_op = att_op
        self.alpha_weight = alpha_weight
        self.out_dim = hidden_size
        if self.att_op == 'concat':
            self.out_dim = hidden_size * 2

        self.emb_encode = nn.Linear(feature_dim_size, hidden_size).double()
        self.dropout = dropout_probability and nn.Dropout(dropout_probability)
        self.z0 = nn.Linear(hidden_size, hidden_size).double()
        self.z1 = nn.Linear(hidden_size, hidden_size).double()
        self.r0 = nn.Linear(hidden_size, hidden_size).double()
        self.r1 = nn.Linear(hidden_size, hidden_size).double()
        self.h0 = nn.Linear(hidden_size, hidden_size).double()
        self.h1 = nn.Linear(hidden_size, hidden_size).double()
        self.soft_att = nn.Linear(hidden_size, 1).double()
        self.ln = nn.Linear(hidden_size, hidden_size).double()
        self.act = act

    def gatedGNN(self, x, adj):
        a = torch.matmul(adj, x)
        # update gate
        z0 = self.z0(a.double())
        z1 = self.z1(x.double())
        z = torch.sigmoid(z0 + z1)
        # reset gate
        r = torch.sigmoid(self.r0(a.double()) + self.r1(x.double()))
        # update embeddings
        h = self.act(self.h0(a.double()) + self.h1(r.double() * x.double()))

        return h * z + x * (1 - z)

    def forward(self, inputs, adj, mask):
        x = self.dropout(inputs) if self.dropout else inputs
        x = self.emb_encode(x.double())
        x = x * mask
        for idx_layer in range(self.num_gnn_layers):
            if self.residual:
                # add residual connection, can use a weighted sum
                x = x + self.gatedGNN(x.double(), adj.double()) * mask.double()
            else:
                x = self.gatedGNN(x.double(), adj.double()) * mask.double()
        # soft attention
        soft_att = torch.sigmoid(self.soft_att(x))
        x = self.act(self.ln(x))
        x = soft_att * x * mask
        # sum/mean and max pooling

        # sum and max pooling
        if self.att_op == 'sum':
            graph_embeddings = torch.sum(x, 1) + torch.amax(x, 1)
        elif self.att_op == 'concat':
            graph_embeddings = torch.cat((torch.sum(x, 1), torch.amax(x, 1)), 1)
        else:
            graph_embeddings = torch.sum(x, 1) * torch.amax(x, 1)

        return graph_embeddings


class ReGCN(nn.Module):
    """GCNs with residual connection"""

    def __init__(
            self,
            feature_dim_size,
            hidden_size,
            num_gnn_layers,
            dropout_probability,
            act=nn.functional.relu,
            residual=True,
            att_op="mul",
            alpha_weight=1.0
    ):
        super(ReGCN, self).__init__()
        self.num_gnn_layers = num_gnn_layers
        self.residual = residual
        self.att_op = att_op
        self.alpha_weight = alpha_weight
        self.out_dim = hidden_size
        if self.att_op == 'concat':
            self.out_dim = hidden_size * 2

        self.gnn_layers = torch.nn.ModuleList()
        for layer in range(self.num_gnn_layers):
            if layer == 0:
                self.gnn_layers.append(GCN(feature_dim_size, hidden_size, dropout_probability, act=act))
            else:
                self.gnn_layers.append(GCN(hidden_size, hidden_size, dropout_probability, act=act))

        self.soft_att = nn.Linear(hidden_size, 1).double()
        self.ln = nn.Linear(hidden_size, hidden_size).double()
        self.act = act

    def forward(self, inputs, adj, mask):
        x = inputs
        for idx_layer in range(self.num_gnn_layers):
            if idx_layer == 0:
                x = self.gnn_layers[idx_layer](x, adj) * mask
            else:
                if self.residual:
                    x = x + self.gnn_layers[idx_layer](x, adj) * mask  # Residual Connection, can use a weighted sum
                else:
                    x = self.gnn_layers[idx_layer](x, adj) * mask
        # soft attention
        soft_att = torch.sigmoid(self.soft_att(x.double()).double())
        x = self.act(self.ln(x))
        x = soft_att * x * mask
        # sum and max pooling
        if self.att_op == 'sum':
            graph_embeddings = torch.sum(x, 1) + torch.amax(x, 1)
        elif self.att_op == 'concat':
            graph_embeddings = torch.cat((torch.sum(x, 1), torch.amax(x, 1)), 1)
        else:
            graph_embeddings = torch.sum(x, 1) * torch.amax(x, 1)

        return graph_embeddings


class GGNN(nn.Module):
    """Gated GNN"""

    def __init__(self, feature_dim_size, hidden_size, num_gnn_layers, dropout_probability, act=nn.functional.relu):
        super(GGNN, self).__init__()
        self.num_gnn_layers = num_gnn_layers
        self.emb_encode = nn.Linear(feature_dim_size, hidden_size).double()
        self.dropout = dropout_probability and nn.Dropout(dropout_probability)
        self.z0 = nn.Linear(hidden_size, hidden_size).double()
        self.z1 = nn.Linear(hidden_size, hidden_size).double()
        self.r0 = nn.Linear(hidden_size, hidden_size).double()
        self.r1 = nn.Linear(hidden_size, hidden_size).double()
        self.h0 = nn.Linear(hidden_size, hidden_size).double()
        self.h1 = nn.Linear(hidden_size, hidden_size).double()
        self.soft_att = nn.Linear(hidden_size, 1).double()
        self.ln = nn.Linear(hidden_size, hidden_size).double()
        self.act = act

    def gatedGNN(self, x, adj):
        a = torch.matmul(adj, x)
        # update gate
        z0 = self.z0(a.double())
        z1 = self.z1(x.double())
        z = torch.sigmoid(z0 + z1)
        # reset gate
        r = torch.sigmoid(self.r0(a.double()) + self.r1(x.double()))
        # update embeddings
        h = self.act(self.h0(a.double()) + self.h1(r.double() * x.double()))

        return h * z + x * (1 - z)

    def forward(self, inputs, adj, mask):
        x = self.dropout(inputs) if self.dropout else inputs
        x = self.emb_encode(x.double())
        x = x * mask
        for idx_layer in range(self.num_gnn_layers):
            x = self.gatedGNN(x.double(), adj.double()) * mask.double()
        return x


class GCN(nn.Module):
    """Simple Graph Convolution layer, similar to https://arxiv.org/abs/1609.02907"""

    def __init__(self, in_features, out_features, dropout_probability, act=torch.relu, bias=False):
        super(GCN, self).__init__()
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.act = act
        self.dropout = dropout_probability and nn.Dropout(dropout_probability)

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(0) + self.weight.size(1)))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, adj):
        x = self.dropout(inputs) if self.dropout else inputs
        support = torch.matmul(x.double(), self.weight.double())
        output = torch.matmul(adj.double(), support.double())
        if self.bias is not None:
            output = output + self.bias
        return self.act(output)
