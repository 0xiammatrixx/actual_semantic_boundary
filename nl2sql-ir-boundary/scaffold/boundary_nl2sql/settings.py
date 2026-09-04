"""The three fixed evaluation settings."""
import random


SETTINGS = {
    "a": dict(n_tables=4, max_cols=3, n_train=4000, n_test=600, tiers=(1, 2),
              para_train=0.4, para_test=0.4, cross_schema=False),
    "b": dict(n_tables=4, max_cols=3, n_train=4000, n_test=600, tiers=(1, 2),
              para_train=0.4, para_test=0.4, cross_schema=True),
    "c": dict(n_tables=4, max_cols=3, n_train=4000, n_test=600, tiers=(1, 2, 3),
              para_train=0.4, para_test=1.0, cross_schema=True),
}


def build_setting(name, seed=0):
    """Return (train_schema, test_schema, train_ex, test_ex, conn, vocab)."""
    import random as _random
    from .data_gen import (make_schema, gen_examples, make_db, build_vocab)

    cfg = SETTINGS[name]
    rng = _random.Random(seed)
    train_schema = make_schema(rng, cfg["n_tables"], cfg["max_cols"])
    # cross-schema: draw a fresh schema with different tables/columns for test
    test_schema = (make_schema(rng, cfg["n_tables"], cfg["max_cols"])
                   if cfg["cross_schema"] else train_schema)
    train_ex = gen_examples(rng, train_schema, cfg["n_train"], cfg["tiers"], cfg["para_train"])
    test_ex = gen_examples(rng, test_schema, cfg["n_test"], cfg["tiers"], cfg["para_test"])
    conn = make_db(rng, test_schema)
    vocab = build_vocab()
    return train_schema, test_schema, train_ex, test_ex, conn, vocab
