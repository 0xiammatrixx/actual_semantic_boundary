"""Deterministic synthetic schema/query/paraphrase generator (fixed).

Generates relational schemas (tables, typed columns, foreign keys), templated
natural-language questions at three complexity tiers, a paraphrase layer, and
SQLite database content so generated SQL can be executed for evaluation.
"""
import random
import re
import sqlite3


TABLES = ["employees", "departments", "projects", "customers",
          "orders", "products", "students", "courses", "instructors", "teams"]
COLS = ["name", "age", "salary", "city", "budget", "location", "grade",
        "credits", "price", "quantity"]
SYNONYMS = {
    "employees": ["staff", "workers", "personnel"],
    "departments": ["divisions", "teams", "units"],
    "projects": ["tasks", "initiatives"],
    "customers": ["clients", "buyers", "patrons"],
    "orders": ["purchases", "sales"],
    "products": ["items", "goods"],
    "students": ["pupils", "learners"],
    "courses": ["classes", "subjects"],
    "name": ["title", "label"],
    "age": ["years", "how old"],
    "salary": ["wage", "pay", "compensation"],
    "city": ["town", "municipality"],
    "budget": ["funding", "allocation"],
    "location": ["place", "site"],
    "grade": ["level", "score"],
    "credits": ["units", "points"],
    "price": ["cost", "rate"],
    "quantity": ["amount", "number"],
}
AGGS = ["none", "count", "sum", "avg", "max", "min"]
CATEGORICAL = ["red", "blue", "green", "alpha", "beta", "gamma"]


def tok(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class Schema:
    def __init__(self, tables, columns, fks):
        self.tables = tables
        self.columns = columns
        self.fks = fks
        self.n_tables = len(tables)
        self.n_cols = len(columns)
        self.table_of_col = [c["table"] for c in columns]
        self.name_of_col = [c["name"] for c in columns]
        self.type_of_col = [c["type"] for c in columns]

    def cols_of(self, t):
        return [i for i in range(self.n_cols) if self.table_of_col[i] == t]


def make_schema(rng, n_tables=4, max_cols=3):
    """Build a schema with globally-unique column names (so lexical match is
    unambiguous on literal questions)."""
    tables = rng.sample(TABLES, n_tables)
    columns, used, pks = [], set(), {}
    for t in range(n_tables):
        pk = tables[t].rstrip("s") + "_id"
        columns.append({"table": t, "name": pk, "type": "INT"})
        used.add(pk); pks[t] = len(columns) - 1
        for cname in rng.sample(COLS, max_cols):
            if cname in used:
                continue
            used.add(cname)
            columns.append({"table": t, "name": cname,
                            "type": "INT" if rng.random() < 0.5 else "TEXT"})
    fks = []
    for t in range(1, n_tables):
        parent = rng.randint(0, t - 1)
        fk = tables[parent].rstrip("s") + "_id"
        child = len(columns)
        columns.append({"table": t, "name": fk, "type": "INT"})
        fks.append((child, pks[parent]))
    return Schema(tables, columns, fks)


def _para(rng, q, prob):
    if prob <= 0:
        return q
    out = []
    for w in q.split(" "):
        core = w.strip(",?.")
        out.append(rng.choice(SYNONYMS[core])
                   if (core in SYNONYMS and rng.random() < prob) else w)
    return " ".join(out)


def gen_examples(rng, schema, n, tiers, para_prob):
    """Generate (question, gold SQL program, gold SQL string) examples."""
    exs = []
    while len(exs) < n:
        tier = rng.choice(tiers)
        t = rng.randrange(schema.n_tables)
        cols = [i for i in schema.cols_of(t) if not schema.name_of_col[i].endswith("_id")]
        if len(cols) < 2:
            continue
        tn = schema.tables[t]
        if tier == 1:
            sel = rng.choice(cols)
            seln = schema.name_of_col[sel]
            q = f"what is the {seln} of {tn}"
            sql = f"SELECT {seln} FROM {tn}"
            gold = dict(table=t, join=None, select=sel, agg="none")
        elif tier == 2:
            grp = rng.choice(cols)
            agg = rng.choice(["count", "sum", "avg", "max", "min"])
            grpn = schema.name_of_col[grp]
            if agg == "count":
                q = f"how many {tn} are there grouped by {grpn}"
                sql = f"SELECT count(*) FROM {tn} GROUP BY {grpn}"
                sel, sel_agg = grp, "count"
            else:
                other = rng.choice([c for c in cols if c != grp])
                on = schema.name_of_col[other]
                q = f"what is the {agg} {on} of {tn} grouped by {grpn}"
                sql = f"SELECT {agg}({on}) FROM {tn} GROUP BY {grpn}"
                sel, sel_agg = other, agg
            gold = dict(table=t, join=None, select=sel, agg=sel_agg)
        else:
            fk_opts = [(c, p) for (c, p) in schema.fks if schema.table_of_col[c] == t]
            if not fk_opts:
                continue
            child, pk = rng.choice(fk_opts)
            t2 = schema.table_of_col[pk]
            t2n = schema.tables[t2]
            seln = schema.name_of_col[child]
            child_name = schema.name_of_col[child]
            pk_name = schema.name_of_col[pk]
            q = f"what is the {seln} of {tn} joined with {t2n}"
            sql = (f"SELECT {tn}.{seln} FROM {tn} JOIN {t2n} ON "
                   f"{tn}.{child_name} = {t2n}.{pk_name}")
            gold = dict(table=t, join=t2, select=child, agg="none")
        q = _para(rng, q, para_prob)
        exs.append(dict(q=q, tokens=tok(q), gold=gold, sql=sql))
    return exs


def make_db(rng, schema, n_rows=12):
    conn = sqlite3.connect(":memory:")
    cur = conn.cursor()
    for t, tname in enumerate(schema.tables):
        cols = schema.cols_of(t)
        defs = ", ".join(f'"{schema.columns[i]["name"]}" '
                         f'{"INTEGER" if schema.columns[i]["type"] == "INT" else "TEXT"}'
                         for i in cols)
        cur.execute(f'CREATE TABLE "{tname}" ({defs})')
        for r in range(n_rows):
            vals = []
            for i in cols:
                c = schema.columns[i]
                if c["name"].endswith("_id"):
                    vals.append(r)
                elif c["type"] == "INT":
                    vals.append(rng.randint(1, 8))
                else:
                    vals.append(rng.choice(CATEGORICAL))
            cur.execute(f'INSERT INTO "{tname}" VALUES ({",".join(["?"] * len(cols))})', vals)
    conn.commit()
    return conn


def execute(sql, conn):
    try:
        cur = conn.cursor()
        cur.execute(sql)
        return tuple(sorted(str(r) for r in cur.fetchall()))
    except Exception:
        return None


def compile_sql(schema, table, sel, agg, join):
    tn = schema.tables[table]
    seln = schema.name_of_col[sel]
    select = seln if agg == "none" else ("count(*)" if agg == "count" else f"{agg}({seln})")
    sql = f"SELECT {select} FROM {tn}"
    if join is not None:
        t2n = schema.tables[join]
        for c, p in schema.fks:
            if schema.table_of_col[c] == table and schema.table_of_col[p] == join:
                sql += f" JOIN {t2n} ON {tn}.{schema.name_of_col[c]} = {t2n}.{schema.name_of_col[p]}"
                break
    if agg not in ("none", "count"):
        sql += f" GROUP BY {seln}"
    return sql


def build_vocab():
    v = {"<pad>": 0, "<unk>": 1}

    def add(w):
        if w not in v:
            v[w] = len(v)

    for w in TABLES + COLS + CATEGORICAL:
        add(w)
    for syns in SYNONYMS.values():
        for w in syns:
            add(w)
    for w in ("what is the of how many are there grouped by joined with and a an to per").split():
        add(w)
    return v
