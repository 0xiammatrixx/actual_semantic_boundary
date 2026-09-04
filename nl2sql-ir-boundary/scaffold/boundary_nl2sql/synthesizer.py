"""Schema linker — the ONLY editable component of this experiment.

The SchemaLinker maps a question + schema into a ``[B, S]`` linking/relevance
matrix that the fixed decoder consumes to select the schema elements (table and
columns) of the generated SQL query. Everything else (encoder, decoder, data,
training, evaluation) is fixed.

The weak baseline (pure lexical string-match) is shipped here. Replace the
``SchemaLinker`` class body with a better linker to reduce the test error.
"""
import torch
import torch.nn as nn


class SchemaGraph:
    """Read-only view of the schema handed to the linker."""

    def __init__(self, schema):
        self.column_names = schema.name_of_col      # list[str]
        self.table_of_column = schema.table_of_col  # list[int] (table index per column)
        self.table_names = schema.tables            # list[str] (table name per index)
        self.fk_edges = schema.fks                 # list[(child_col, parent_col)]
        self.n_columns = schema.n_cols


# ---------------------------------------------------------------------------
# EDITABLE RANGE: the `SchemaLinker` class.
#
# It must be a torch.nn.Module whose `build` method returns a [B, S] tensor of
# linking/relevance scores, where B is the batch size and S is the number of
# schema columns. The fixed decoder selects the highest-scoring column/table.
# ---------------------------------------------------------------------------
class SchemaLinker(nn.Module):
    """Returns [B, S] linking scores between question and schema columns.

    Args:
        q_sum:         [B, d] question summary (mean-pooled encoder output)
        schema_repr:   [S, d] column-name embeddings from the fixed encoder
        schema_graph:  SchemaGraph (column names, table membership, fk edges,
                       table names)
        lexical:       [B, S] binary exact string-match matrix (question token
                       == column-name token)
        q_repr:        [B, L, d] per-token question embeddings from the encoder
        q_mask:        [B, L] float mask (1 for real tokens, 0 for padding)
        table_lexical: [B, T] binary exact string-match matrix (question token
                       == table-name token)
    """

    def __init__(self, d):
        super().__init__()

    def build(self, q_sum, schema_repr, schema_graph, lexical, q_repr, q_mask,
              table_lexical):
        return lexical  # weak baseline: pure lexical string-match
