class SchemaLinker(nn.Module):
    """Relation-aware linking: table-membership boost + lexical fusion.

    Reference: Wang, Shin, Liu, Polozov & Richardson (2020), *RAT-SQL:
    Relation-Aware Schema Encoding and Linking for Text-to-SQL Parsers* (ACL),
    whose linking module combines exact matching with learned attention and
    exploits schema structure; and Guo, Zhan, Gao, Lou, Liu & Zhang (2019),
    *Towards Complex Text-to-SQL in Cross-Domain Database with Intermediate
    Representation* (ACL), which fuses multiple schema-linking signals.

    Uses the schema graph's table membership to boost every column of a table
    whose name appears in the question, so a cross-table homonym is resolved by
    table context. Adds the exact string-match matrix as well, so literal
    questions are resolved exactly while paraphrased ones are handled by the
    learned bilinear attention.
    """

    def __init__(self, d):
        super().__init__()
        self.W = nn.Parameter(torch.eye(d))
        self.w_lex = nn.Parameter(torch.tensor(1.0))
        self.w_table = nn.Parameter(torch.tensor(3.0))

    def build(self, q_sum, schema_repr, schema_graph, lexical, q_repr, q_mask,
              table_lexical):
        toc = torch.tensor(schema_graph.table_of_column, device=schema_repr.device)
        table_boost = table_lexical[:, toc]  # [B, S]: 1 if the column's table is named
        return (q_sum @ self.W @ schema_repr.transpose(0, 1)
                + self.w_lex * lexical
                + self.w_table * table_boost)
