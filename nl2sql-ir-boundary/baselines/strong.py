class SchemaLinker(nn.Module):
    """Lexical + learned soft-attention fusion linking.

    Reference: Guo, Zhan, Gao, Lou, Liu & Zhang (2019), *Towards Complex
    Text-to-SQL in Cross-Domain Database with Intermediate Representation*
    (ACL), which fuses multiple schema-linking signals; and Wang, Shin, Liu,
    Polozov & Richardson (2020), *RAT-SQL: Relation-Aware Schema Encoding and
    Linking for Text-to-SQL Parsers* (ACL), whose linking module combines
    n-gram exact matching with learned attention.

    Adds a strongly weighted exact string-match matrix to a trainable bilinear
    score q^T W s between the question summary and each column embedding, so
    literal questions are resolved exactly while paraphrased ones are handled
    by the learned attention.
    """

    def __init__(self, d):
        super().__init__()
        self.W = nn.Parameter(torch.eye(d))

    def build(self, q_sum, schema_repr, schema_graph, lexical, q_repr, q_mask):
        return q_sum @ self.W @ schema_repr.transpose(0, 1) + 5.0 * lexical
