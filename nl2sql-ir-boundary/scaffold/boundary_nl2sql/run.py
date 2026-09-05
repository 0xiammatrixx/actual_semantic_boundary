"""Entry point. Each "setting" is a fixed evaluation command that runs this
module, trains the fixed NL->SQL model, and prints one `TEST_METRICS:` line
(primary metric first, secondary metric second).

Example (visible settings):
    python -m boundary_nl2sql.run --setting a
    python -m boundary_nl2sql.run --setting b
"""
import argparse

from .settings import build_setting
from .train import train_and_eval


def main():
    p = argparse.ArgumentParser(description="Train the fixed NL->SQL model.")
    p.add_argument("--setting", choices=["a", "b", "c"], default="a")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=25)
    args = p.parse_args()

    train_schema, test_schema, train_ex, test_ex, conn, vocab = build_setting(
        args.setting, args.seed)

    col_acc, tab_acc, cm, ea = train_and_eval(
        train_ex, test_ex, train_schema, test_schema, conn, vocab,
        seed=args.seed, epochs=args.epochs)

    # Primary metric (execution accuracy) first, secondary (component match) second.
    print(f"TEST_METRICS: {ea:.6f} {cm:.6f}")


if __name__ == "__main__":
    main()
