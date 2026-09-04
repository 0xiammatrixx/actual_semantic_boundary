"""Fixed decoder (do not modify). Consumes the linker's [B, S] relevance."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, d, n_tables, n_aggs):
        super().__init__()
        self.agg_head = nn.Linear(d, n_aggs)

    def forward(self, relevance, q_sum, table_of_col, n_tables):
        # relevance: [B, S] linking scores -> column logits
        col_logits = relevance
        mask = F.one_hot(table_of_col, n_tables).bool().unsqueeze(0)
        table_logits = col_logits.unsqueeze(2).masked_fill(~mask, -1e9).max(dim=1).values
        return col_logits, table_logits, self.agg_head(q_sum)
