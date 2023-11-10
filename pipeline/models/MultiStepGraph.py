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

    def forward(self, graph_emb, ptr, batch):
        n_nodes = ptr[1:] - ptr[:-1]
        max_node = n_nodes.max().item() + 1
        batch_size = batch.max().item() + 1
        device = graph_emb.device
        batch_mask = torch.zeros(batch_size, max_node).to(device)
        all_feats = torch.zeros(batch_size, max_node, self.emb_dim).to(device)

        for idx, x in enumerate(n_nodes):
            batch_mask[idx, :x.item()] = True

        all_feats[batch_mask] = graph_emb

        attn_o, attn_w = self.pooler(
            query=self.qry_embedding.repeat(batch_size, 1, 1),
            key=all_feats, value=all_feats, key_padding_mask=~batch_mask
        )
        return attn_o.squeeze(dim=1)

