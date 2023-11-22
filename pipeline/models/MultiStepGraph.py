from torch_geometric.nn.inits import glorot, zeros
import torch_geometric
from torch.nn import Parameter
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import MessagePassing
import torch.nn.functional as F
from torch_geometric.utils import softmax as sp_softmax
import torch


class AttentionPooling(torch.nn.Module):
    def __init__(
        self, output_dim, num_heads, emb_dim=None, dropout=0.1
    ):
        super(AttentionPooling, self).__init__()
        self.qry_embedding = torch.nn.Parameter(torch.randn(1, 1, output_dim))
        self.pooler = torch.nn.MultiheadAttention(
            output_dim, num_heads=num_heads, batch_first=True,
            kdim=emb_dim, vdim=emb_dim, dropout=dropout
        )
        self.output_dim = output_dim
        self.emb_dim = output_dim if emb_dim is None else emb_dim

    def forward(self, graph_emb, batch):
        batch_size = batch.max().item() + 1
        n_nodes = torch.zeros(batch_size).to(batch)
        n_nodes.index_add_(torch.ones_like(batch), index=batch, dim=0)
        max_node = n_nodes.max().item() + 1

        device = graph_emb.device
        batch_mask = torch.zeros(batch_size, max_node).bool().to(device)
        all_feats = torch.zeros(batch_size, max_node, self.emb_dim).to(device)

        for idx, x in enumerate(n_nodes):
            batch_mask[idx, :x.item()] = True

        all_feats[batch_mask] = graph_emb

        attn_o, attn_w = self.pooler(
            query=self.qry_embedding.repeat(batch_size, 1, 1),
            key=all_feats, value=all_feats, key_padding_mask=~batch_mask
        )
        return attn_o.squeeze(dim=1)


class SelfLoopGATConv(MessagePassing):
    def __init__(
        self, in_channels, out_channels, edge_dim, heads=1,
        negative_slope=0.2, dropout=0.1, **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super(SelfLoopGATConv, self).__init__(node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.edge_dim = edge_dim
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.dropout_fun = torch.nn.Dropout(dropout)

        self.lin_src = self.lin_dst = Linear(
            in_channels, out_channels * heads,
            bias=False, weight_initializer='glorot'
        )

        self.att_src = Parameter(torch.zeros(1, heads, out_channels))
        self.att_dst = Parameter(torch.zeros(1, heads, out_channels))
        self.att_edge = Parameter(torch.zeros(1, heads, out_channels))

        self.bias = Parameter(torch.zeros(heads * out_channels))
        self.lin_edge = Linear(
            edge_dim, out_channels * heads,
            bias=True, weight_initializer='glorot'
        )
        self.self_edge = torch.nn.Parameter(torch.randn(1, edge_dim))
        self.reset_parameters()

    def reset_parameters(self):
        if torch_geometric.__version__.startswith('2.3'):
            super(SelfLoopGATConv, self).reset_parameters()
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        self.lin_edge.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        glorot(self.att_edge)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr, size=None):
        num_nodes = x.shape[0]

        # add self loop

        self_edges = torch.Tensor([(i, i) for i in range(num_nodes)])
        self_edges = self_edges.T.to(edge_index)

        edge_index = torch.cat([edge_index, self_edges], dim=1)
        real_edge_attr = torch.cat([
            edge_attr, self.self_edge.repeat(num_nodes, 1)
        ], dim=0)

        # old prop

        H, C = self.heads, self.out_channels
        x_src = self.lin_src(x).view(-1, H, C)
        x_dst = self.lin_dst(x).view(-1, H, C)
        edge_attr = self.lin_edge(real_edge_attr)

        x = (x_src, x_dst)
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = (x_dst * self.att_dst).sum(dim=-1)
        alpha = (alpha_src, alpha_dst)

        alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr)

        out = self.propagate(
            edge_index, x=x, alpha=alpha, size=size,
            edge_attr=edge_attr.view(-1, H, C)
        )
        out = out.view(-1, H * C) + self.bias
        return out

    def edge_update(self, alpha_j, alpha_i, edge_attr, index, ptr, size_i):
        edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
        alpha_edge = (edge_attr * self.att_edge).sum(dim=-1)
        alpha = alpha_i + alpha_j + alpha_edge

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = self.dropout_fun(sp_softmax(alpha, index, ptr, size_i))
        return alpha

    def message(self, x_j, alpha, edge_attr):
        return alpha.unsqueeze(-1) * (x_j + edge_attr)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channels}, '
            f'{self.out_channels}, heads={self.heads})'
        )


class RXNGAT(torch.nn.Module):
    def __init__(
        self, n_layer, emb_dim, gnn_dim, negative_slope=0.2,
        dropout=0.1, heads=1, component_keys={'reactants', 'products'}
    ):
        super(RXNGAT, self).__init__()
        self.n_layer = n_layer
        self.emb_dim = emb_dim
        self.gnn_dim = gnn_dim

        self.from_bond_embs = torch.nn.ParameterDict({
            k: torch.nn.Parameter(torch.randn(1, emb_dim))
            for k in component_keys
        })
        self.to_bond_embs = torch.nn.ParameterDict({
            k: torch.nn.Parameter(torch.randn(1, emb_dim))
            for k in component_keys
        })
        self.component_keys = component_keys

        self.Attn_pools = torch.nn.ModuleDict({
            k: AttentionPooling(
                output_dim=emb_dim, num_heads=heads,
                dropout=dropout, emb_dim=gnn_dim
            ) for k in component_keys
        })
        self.rxn_key_emb = torch.nn.Parameter(torch.randn(emb_dim))

        assert emb_dim % heads == 0, \
            'Number of hidden should be evenly divided by heads'

        self.layers = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        for idx in range(n_layer):
            self.layers.append(SelfLoopGATConv(
                emb_dim, emb_dim // heads, emb_dim, heads=heads,
                dropout=dropout, negative_slope=negative_slope
            ))
            self.lns.append(torch.nn.LayerNorm(emb_dim))
        self.dropout_fun = torch.nn.Dropout(dropout)

    def make_graph(self, graph_with_embs):
        # graph pooling
        batch_size = None
        key2emb = {}
        for key, val in graph_with_embs.items():
            this_emb, this_graph = val['embeddings'], val['graph']
            pooled_emb = self.Attn_pools[key](this_emb, this_graph.batch)
            key2emb[key] = pooled_emb
            if batch_size is None:
                batch_size = pooled_emb.shape[0]
            assert batch_size == pooled_emb.shape[0], \
                f"batch size of {key} should be {batch_size}" +\
                f' but {pooled_emb.shape[0]} Found!'
            device = pooled_emb.device

        # graph definition

        node_per_graph = len(graph_with_embs) + 1
        whole_x = torch.zeros(batch_size * node_per_graph)
        whole_x = whole_x.to(device)
        whole_edge_attr, whole_eidx = [], []

        key2idx = {}

        for idx, key in enumerate(graph_with_embs.keys()):
            x_idx = [i * node_per_graph + idx for i in range(batch_size)]
            key2idx[key], whole_x[x_idx] = x_idx, key2emb[key]

            from_emb = self.from_bond_embs[key].repeat(batch_size, 1)
            to_emb = self.to_bond_embs[key].repeat(batch_size, 1)
            from_edges = [
                [i * node_per_graph + idx, (i + 1) * node_per_graph - 1]
                for i in range(batch_size)
            ]
            to_edges = [
                [(i + 1) * node_per_graph - 1, i * node_per_graph + idx]
                for i in range(batch_size)
            ]

            whole_eidx.extend(from_edges)
            whole_edge_attr.append(from_emb)
            whole_eidx.extend(to_edges)
            whole_edge_attr.append(to_emb)

        # rxns

        x_idx = [(i + 1) * node_per_graph - 1 for i in range(batch_size)]
        key2idx['rxn'], whole_x[x_idx] = x_idx, self.rxn_key_emb

        batch, ptr = [], [0]
        for i in range(batch_size):
            batch.extend([i] * node_per_graph)
            ptr.append((i + 1) * node_per_graph)

        batch = torch.LongTensor(batch).to(device)
        prt = torch.LongTensor(ptr).to(device)

        return torch_geometric.data.Data(**{
            'x': whole_x, 'batch': batch, 'ptr': ptr,
            'num_nodes': batch_size * node_per_graph,
            'edge_attr': torch.cat(whole_edge_attr, dim=0),
            'edge_index': torch.LongTensor(whole_eidx).to(device).T
        }), key2idx

    def forward(self, graph_with_embs):
        rxn_graph, key2idx = self.make_graph(graph_with_embs)

        x = rxn_graph.x
        for i in range(self.n_layer):
            x = self.lns[i](self.layers[i](
                x=x, edge_index=rxn_graph.edge_index,
                edge_attr=rxn_graph.edge_attr
            ) + x)

            x = torch.relu(self.dropout_fun(x))

        return {k: x[v] for k, v in key2idx.items()}
