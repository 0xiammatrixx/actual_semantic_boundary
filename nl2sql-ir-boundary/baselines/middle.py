class SchemaLinker(nn.Module):
    """Learned bilinear soft-attention linking.

    Reference: Xu, Liu & Song (2017), *SQLNet: Generating Structured Queries From
    Natural Language Without Reinforcement Learning* (arXiv:1711.04436).

    A trainable bilinear score q^T W s between the question summary and each
    column embedding. Learns from data but ignores schema graph structure
    (foreign keys, table membership).
    """

    def __init__(self, d):
        super().__init__()
        self.W = nn.Parameter(torch.eye(d))

    def build(self, q_sum, schema_repr, schema_graph, lexical, q_repr, q_mask):
        return q_sum @ self.W @ schema_repr.transpose(0, 1)
