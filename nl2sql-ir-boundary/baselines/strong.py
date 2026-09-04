class SchemaLinker(nn.Module):
    """Relation-aware linking (schema-graph encoding + lexical fusion).

    Reference: Wang, Shin, Liu, Polozov & Richardson (2020), *RAT-SQL:
    Relation-Aware Schema Encoding and Linking for Text-to-SQL Parsers* (ACL).

    Propagates column embeddings over foreign-key and same-table edges (a
    relation-aware schema encoder) before the same bilinear attention as the
    middle baseline, then adds the exact string-match matrix so literal
    questions are resolved exactly while paraphrased ones are handled by the
    learned, structure-aware attention.
    """

    def __init__(self, d):
        super().__init__()
        self.W = nn.Parameter(torch.eye(d))
        self.gnn = nn.Linear(d, d)
        nn.init.normal_(self.gnn.weight, std=0.01)
        nn.init.zeros_(self.gnn.bias)

    def build(self, q_sum, schema_repr, schema_graph, lexical):
        S = schema_repr.shape[0]
        table = torch.tensor(schema_graph.table_of_column, device=schema_repr.device)
        adj = (table.unsqueeze(0) == table.unsqueeze(1)).float()
        adj = adj + torch.eye(S, device=schema_repr.device)
        for c, p in schema_graph.fk_edges:
            adj[c, p] = 1.0
            adj[p, c] = 1.0
        adj = adj / adj.sum(dim=1, keepdim=True).clamp(min=1)
        prop = torch.relu(self.gnn(adj @ schema_repr))
        schema_repr2 = schema_repr + 0.1 * prop
        return q_sum @ self.W @ schema_repr2.transpose(0, 1) + lexical
