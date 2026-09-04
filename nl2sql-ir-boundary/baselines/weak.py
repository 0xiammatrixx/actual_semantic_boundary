class SchemaLinker(nn.Module):
    """Lexical string-match linking.

    Reference: the heuristic schema-linking step described in Yu et al. (2018),
    *SyntaxSQLNet: Syntax Tree Networks for Complex and Cross-Domain Text-to-SQL
    Task Generation* (EMNLP), and Guo et al. (2019), *Towards Complex Text-to-SQL
    in Cross-Domain Database with Intermediate Representation* (ACL).

    Aligns question tokens to schema columns by exact string overlap, with no
    learned parameters. Fails when the question paraphrases column names.
    """

    def __init__(self, d):
        super().__init__()

    def build(self, q_sum, schema_repr, schema_graph, lexical, q_repr, q_mask,
              table_lexical):
        return lexical
