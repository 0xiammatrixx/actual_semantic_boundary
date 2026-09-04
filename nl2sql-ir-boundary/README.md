# nl2sql-ir-boundary — Hearts task

**Research question:** Does a relation-aware, schema-graph-structured linking
operator produce more accurate and more executable SQL than lexical or purely
learned soft-attention linking, under cross-schema and linguistic-paraphrase
distribution shift?

This task asks an agent to improve a single, well-isolated component — the
schema linker that aligns a natural-language question to database schema
columns — while everything else (encoder, decoder, data generator, optimizer,
training budget, evaluation) stays frozen.

## Reproduced baselines (weakest → strongest)

All three are reproduced under the *same* seed, data, encoder, decoder, and
training budget; they differ only in the body of the editable `SchemaLinker`
class in `scaffold/boundary_nl2sql/synthesizer.py`.

| ID | Method | Citation | Notes |
|----|--------|----------|-------|
| `weak` | Lexical string-match linking | Yu, Li, Gao, Xie, Mou, Pang, Song (2018), *SyntaxSQLNet: Syntax Tree Networks for Complex and Cross-Domain Text-to-SQL Task* (EMNLP); Guo, Zhan, Gao, Lou, Liu, Zhang (2019), *Towards Complex Text-to-SQL in Cross-Domain Database with Intermediate Representation* (ACL). | Exact token overlap between question and column names; no learned parameters. Collapses when the question paraphrases column names. |
| `middle` | Learned bilinear soft-attention | Xu, Liu & Song (2017), *SQLNet: Generating Structured Queries From Natural Language Without Reinforcement Learning* (arXiv:1711.04436). | Trainable bilinear score $q^\top W s$ between question summary and each column embedding. Learns from data but ignores schema graph structure (foreign keys, table membership). |
| `strong` | Relation-aware linking (attention + schema-graph encoding) | Wang, Shin, Liu, Polozov & Richardson (2020), *RAT-SQL: Relation-Aware Schema Encoding and Linking for Text-to-SQL Parsers* (ACL). | Propagates column embeddings over foreign-key and same-table edges before the same bilinear attention, so linking is aware of schema structure. |

The **shipped scaffold** uses `weak` (lexical string-match). The **oracle**
(score to beat) is `strong` (relation-aware). A passing trial must raise
execution accuracy *above* the relation-aware composite anchor.

## Experiment

- **Encoder (fixed):** word + column-name BiLSTM (`boundary_nl2sql/encoder.py`).
- **Decoder (fixed):** selects table / SELECT column / aggregation from the
  linker's relevance scores (`boundary_nl2sql/decoder.py`).
- **Data (fixed, deterministic):** synthetic schemas with unique column names,
  questions at three complexity tiers (select / aggregate-groupby / join), and
  a paraphrase layer (`boundary_nl2sql/data_gen.py`). SQLite executes the SQL.
- **Training (fixed):** 25 epochs, learning rate 1e-3, batch 64, seed 0,
  cross-entropy over table/column/aggregation (`boundary_nl2sql/train.py`).
- **Metric:** primary = execution accuracy (higher better); secondary =
  component match (higher better). Printed as `TEST_METRICS: <ea> <cm>`.

### Settings

| Name | Visible | Command | Shift |
|------|---------|---------|-------|
| `a` | yes | `python -m boundary_nl2sql.run --setting a` | in-schema, 40% paraphrase, tiers 1–2 |
| `b` | yes | `python -m boundary_nl2sql.run --setting b` | cross-schema (fresh test schema), 40% paraphrase, tiers 1–2 |
| `c` | no (hidden) | `python -m boundary_nl2sql.run --setting c` | cross-schema + 100% paraphrase + join tier 3 |

The hidden setting `c` combines a novel schema, full paraphrase, and join
queries — exactly the conditions where lexical matching collapses and
schema-structure awareness matters most.

## Reproducibility

- Base image `python:3.12-slim`; `torch==2.2.2+cpu` (CPU-only), `numpy==1.26.4`.
- Seeds fixed (`--seed 0`; data and model seeds fixed at 0).
- Single-threaded math (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`).
- Hardware class: CPU (no GPU).

## Results (measured with the fixed seed)

See `leaderboard.csv`. One aggregate row per baseline across the three settings.

## Reduced scale

The experiment is intentionally small so each setting finishes in tens of
minutes on CPU: 4 000 train / 600 test questions, embedding dim 96, 25 epochs.
This preserves the relative ordering of the three baselines while keeping the
full end-to-end run fast.

## Source

This task, its scaffold, data generator, evaluation setup, and hidden shift are
original. The three methods above are published baselines reproduced inside the
editable range; their implementations are small self-contained PyTorch modules
(see `baselines/`).
