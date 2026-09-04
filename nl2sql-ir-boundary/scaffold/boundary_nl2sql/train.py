"""Fixed training loop + evaluation (do not modify).

The encoder, decoder, optimizer, seed, and evaluation are fixed. Only the
`SchemaLinker` (imported from `synthesizer.py`) varies across baselines/trials.
"""
import random

import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder
from .synthesizer import SchemaLinker, SchemaGraph
from .data_gen import AGGS, compile_sql, execute, tok


def _lexical(ids, match_lookup):
    # ids: [B, L] token ids; match_lookup: [V, S] binary. -> [B, S]
    m = match_lookup[ids]
    return m.max(dim=1).values


def train_and_eval(train_ex, test_ex, train_schema, test_schema, conn, vocab,
                   d=96, n_tables=None, epochs=25, batch=64, lr=1e-3, seed=0):
    if n_tables is None:
        n_tables = train_schema.n_tables
    torch.manual_seed(seed)
    inv = {v: k for k, v in vocab.items()}

    n_aggs = len(AGGS)
    enc = Encoder(len(vocab), d, d // 2)
    linker = SchemaLinker(d)
    dec = Decoder(enc.out_dim, n_tables, n_aggs)

    opt = torch.optim.Adam(
        list(enc.parameters()) + list(dec.parameters()) + list(linker.parameters()),
        lr=lr)

    def schema_tensors(schema):
        col_name_ids = nn.utils.rnn.pad_sequence(
            [torch.tensor([vocab.get(t, 1) for t in tok(n)]) for n in schema.name_of_col],
            batch_first=True)
        table_of_col = torch.tensor(schema.table_of_col)
        names_tok = [tok(n) for n in schema.name_of_col]
        M = torch.zeros(len(vocab), schema.n_cols)
        for vidx in range(len(vocab)):
            w = inv.get(vidx, "")
            if not w:
                continue
            for s, nt in enumerate(names_tok):
                if w in nt:
                    M[vidx, s] = 1.0
        return col_name_ids, table_of_col, M

    train_ids, train_toc, train_ml = schema_tensors(train_schema)
    test_ids, test_toc, test_ml = schema_tensors(test_schema)
    train_sg = SchemaGraph(train_schema)
    test_sg = SchemaGraph(test_schema)

    def forward(exs, schema_repr, ml, sg, toc):
        ids = nn.utils.rnn.pad_sequence(
            [torch.tensor([vocab.get(t, 1) for t in e["tokens"]]) for e in exs],
            batch_first=True)
        lens = torch.tensor([len(e["tokens"]) for e in exs])
        q_repr = enc.encode(ids, lens)
        q_mask = (ids != 0).float()
        q_sum = (q_repr * q_mask.unsqueeze(2)).sum(1) / q_mask.sum(1, keepdim=True).clamp(min=1)
        lexical = _lexical(ids, ml)
        relevance = linker.build(q_sum, schema_repr, sg, lexical, q_repr, q_mask)
        return dec(relevance, q_sum, toc, n_tables)

    lossfn = nn.CrossEntropyLoss()

    rng = random.Random(123)
    for _ in range(epochs):
        rng.shuffle(train_ex)
        enc.train(); dec.train(); linker.train()
        for i in range(0, len(train_ex), batch):
            exs = train_ex[i:i + batch]
            s_repr = enc.encode_names(train_ids)
            col_logits, table_logits, agg_logits = forward(exs, s_repr, train_ml, train_sg, train_toc)
            g = [e["gold"] for e in exs]
            loss = (lossfn(table_logits, torch.tensor([x["table"] for x in g]))
                    + lossfn(col_logits, torch.tensor([x["select"] for x in g]))
                    + lossfn(agg_logits, torch.tensor([AGGS.index(x["agg"]) for x in g])))
            opt.zero_grad()
            loss.backward()
            opt.step()

    enc.eval(); dec.eval(); linker.eval()
    col_acc = tab_acc = cm = ea = total = 0
    with torch.no_grad():
        s_repr = enc.encode_names(test_ids)
        for i in range(0, len(test_ex), 64):
            exs = test_ex[i:i + 64]
            col_logits, table_logits, agg_logits = forward(exs, s_repr, test_ml, test_sg, test_toc)
            ps, pt, pa = col_logits.argmax(1), table_logits.argmax(1), agg_logits.argmax(1)
            for j, ex in enumerate(exs):
                g = ex["gold"]
                col_acc += ps[j].item() == g["select"]
                tab_acc += pt[j].item() == g["table"]
                cm += (pt[j].item() == g["table"] and ps[j].item() == g["select"]
                       and pa[j].item() == AGGS.index(g["agg"]))
                pred_sql = compile_sql(test_schema, pt[j].item(), ps[j].item(),
                                       AGGS[pa[j].item()], g["join"], g.get("meas"))
                gold_res = execute(ex["sql"], conn)
                pred_res = execute(pred_sql, conn)
                ea += (gold_res is not None and pred_res == gold_res)
                total += 1
    return col_acc / total, tab_acc / total, cm / total, ea / total
