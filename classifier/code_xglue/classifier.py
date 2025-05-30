import torch.nn as nn
import torch
import torch.nn.functional as F

from layout import ReGCN, ReGGNN, PredictionLayout
from graph import build_graph, build_graph_text
from preprocess import preprocess_features, preprocess_adj


class DefaultClassifier(nn.Module):
    def __init__(self, encoder, tokenizer, dropout_probability=0):
        super(DefaultClassifier, self).__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer

        self.dropout = dropout_probability > 0 and nn.Dropout(dropout_probability)

    def forward(self, inputs=None, labels=None):
        outputs = self.encoder(inputs, attention_mask=inputs.ne(1))[0]
        logits = self.dropout(outputs) if self.dropout else outputs

        p = torch.sigmoid(logits)
        if labels is None:
            return p

        labels = labels.float()
        loss = torch.log(p[:, 0] + 1e-10) * labels + torch.log((1 - p)[:, 0] + 1e-10) * (1 - labels)
        loss = -loss.mean()
        return loss, p


class GNNReGVD(nn.Module):
    def __init__(
            self, encoder, tokenizer,
            transformer,
            device, layout,
            dropout_probability=0,
            hidden_size=256,
            feature_dim_size=768,
            num_gnn_layers=2,
            remove_residual=False,
            att_op='mul',
            from_text=False
    ):
        super(GNNReGVD, self).__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer

        self.word_embeddings = transformer.embeddings.word_embeddings.weight.data.cpu().detach().clone().numpy()

        self.device = device
        if layout == "ReGGNN":
            GNN = ReGGNN
        elif layout == "ReGCN":
            GNN = ReGCN
        else:
            raise NotImplementedError(layout)

        self.gnn = GNN(
            feature_dim_size=feature_dim_size,
            hidden_size=hidden_size,
            num_gnn_layers=num_gnn_layers,
            dropout_probability=dropout_probability,
            residual=not remove_residual,
            att_op=att_op
        )
        self.build_graph = build_graph_text if from_text else build_graph
        self.predictor = PredictionLayout(input_size=self.gnn.out_dim)

    def forward(self, input_ids=None, labels=None):
        # construct graph
        adj, x_feature = self.build_graph(
            input_ids.cpu().detach().numpy(),
            self.word_embeddings
        )

        # initialization
        adj, adj_mask = preprocess_adj(adj)
        adj_feature = preprocess_features(x_feature)
        adj = torch.from_numpy(adj)
        adj_mask = torch.from_numpy(adj_mask)
        adj_feature = torch.from_numpy(adj_feature)
        # run over GNNs
        outputs = self.gnn(
            adj_feature.to(self.device).double(),
            adj.to(self.device).double(),
            adj_mask.to(self.device).double()
        )
        logits = self.predictor(outputs)
        prob = F.sigmoid(logits)
        if labels is None:
            return prob

        labels = labels.float()
        loss = torch.log(prob[:, 0] + 1e-10) * labels + torch.log((1 - prob)[:, 0] + 1e-10) * (1 - labels)
        loss = -loss.mean()
        return loss, prob
