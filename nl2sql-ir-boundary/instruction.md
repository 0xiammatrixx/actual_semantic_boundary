# Improve the schema linker of a fixed text-to-SQL system

## Task

You are given a fixed, deterministic PyTorch experiment that maps natural
language questions to SQL queries against a database schema, and reports how
well the generated SQL executes.

The system currently uses a **pure lexical string-match** schema linker: it
aligns question tokens to schema columns only when the column name appears
verbatim in the question. This works on literal questions but fails when the
question paraphrases the column/table names. Your job is to replace the linker
with a better one to **increase execution accuracy**.

## What you may change

Exactly one code block: the `SchemaLinker` class in
`boundary_nl2sql/synthesizer.py`. Its `build` method must return a `[B, S]`
tensor of linking/relevance scores (B = batch size, S = number of schema
columns). The fixed decoder selects the highest-scoring column and table to
construct the SQL query.

The `build` method receives:

- `q_sum` `[B, d]` — the mean-pooled question summary from the fixed encoder.
- `schema_repr` `[S, d]` — the column-name embeddings from the fixed encoder.
- `schema_graph` — read-only schema view: column names, table membership
  (`table_of_column`), and foreign-key edges (`fk_edges`).
- `lexical` `[B, S]` — binary exact string-match matrix (question token ==
  column-name token).
- `q_repr` `[B, L, d]` — per-token question embeddings from the fixed encoder.
- `q_mask` `[B, L]` — float mask (1 for real tokens, 0 for padding).

The decoder selects three things from the linker's scores and the question:

- **table** — the database table of the query (highest-scoring table).
- **column** — the highest-scoring schema column. What this column means
  depends on the question tier: for a plain select it is the `SELECT` column;
  for an aggregate/group-by question it is the **`GROUP BY` column** (the
  aggregated *measure* column is fixed per table); for a join question it is a
  regular `SELECT` column of the named table (the join target is given).
- **aggregation** — `none` / `count` / `sum` / `avg` / `max` / `min`.

You may **not** change anything else: the encoder, the decoder, the optimizer,
the learning rate, the number of epochs, the data generator, or the evaluator.

## What stays fixed

- **Encoder**: BiLSTM word/column-name encoder (`boundary_nl2sql/encoder.py`).
- **Decoder**: uses the linker's scores to pick table/column/aggregation
  (`boundary_nl2sql/decoder.py`).
- **Data**: deterministic synthetic schemas, questions (3 complexity tiers),
  and a paraphrase layer (`boundary_nl2sql/data_gen.py`).
- **Training**: fixed optimizer, 25 epochs, seed 0 (`boundary_nl2sql/train.py`).
- **Metrics**: execution accuracy (primary, higher is better) and component
  match (secondary, higher is better).

## How to run

From the `scaffold/` directory:

```bash
python -m boundary_nl2sql.run --setting a
```

This trains the model and prints one line:

```
TEST_METRICS: <execution_accuracy> <component_match>
```

A second, cross-schema visible setting can be evaluated with:

```bash
python -m boundary_nl2sql.run --setting b
```

## Goal

Increase execution accuracy as much as possible by choosing a better schema
linker. Your final linker must generalize across settings (including schemas
and phrasing you may not have seen), not just perform well on a single setting.
