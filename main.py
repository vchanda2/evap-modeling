"""
CLI entry point for training and evaluating predictors.

Usage examples
--------------
# Train the decision tree predictor and save it
python main.py train --predictor dt --seq-length 6 --n-steps 12 --save models/dt.pkl

# Load a saved predictor and print a quick prediction summary
python main.py predict --predictor dt --load models/dt.pkl
"""

import argparse
from pathlib import Path

from src.models.system import RefrigerationSystem
from src.predictors.decision_tree_predictor import DecisionTreePredictor
from src.predictors.lstm_predictor import LSTMPredictor
from src.config import DEFAULT_SEQ_LENGTH, DEFAULT_N_STEPS


PREDICTORS = {
    "dt": DecisionTreePredictor,
    "lstm": LSTMPredictor,
}


def build_predictor(name: str, seq_length: int, n_steps: int):
    cls = PREDICTORS.get(name)
    if cls is None:
        raise ValueError(f"Unknown predictor '{name}'. Choose from: {list(PREDICTORS)}")
    return cls(seq_length=seq_length, n_steps=n_steps)


def cmd_train(args):
    print(f"Loading system data...")
    system = RefrigerationSystem()
    system.load()
    print(system)

    predictor = build_predictor(args.predictor, args.seq_length, args.n_steps)
    print(f"Training {predictor.__class__.__name__}...")
    predictor.fit(system)

    if args.save:
        predictor.save(Path(args.save))


def cmd_predict(args):
    import numpy as np

    predictor = build_predictor(args.predictor, args.seq_length, args.n_steps)
    predictor.load(Path(args.load))

    system = RefrigerationSystem()
    system.load()

    df = system.get_dataframe()
    input_cols = system.input_feature_cols
    output_cols = system.output_feature_cols

    # Take the last seq_length rows as the prediction window
    window = df[input_cols].iloc[-predictor.seq_length:].to_numpy(dtype="float32")
    predictions = predictor.predict(window)  # (n_steps, n_out)

    print(f"\nPredicted next {predictor.n_steps} steps:")
    print(f"{'Step':>5}  " + "  ".join(f"{c:>20}" for c in output_cols))
    for i, row in enumerate(predictions):
        print(f"{i+1:>5}  " + "  ".join(f"{v:>20.3f}" for v in row))


def main():
    parser = argparse.ArgumentParser(description="Evaporator modeling CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # -- train --
    train_p = sub.add_parser("train", help="Train a predictor")
    train_p.add_argument("--predictor", choices=list(PREDICTORS), default="dt")
    train_p.add_argument("--seq-length", type=int, default=DEFAULT_SEQ_LENGTH)
    train_p.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS)
    train_p.add_argument("--save", type=str, default=None, help="Path to save model")
    train_p.set_defaults(func=cmd_train)

    # -- predict --
    pred_p = sub.add_parser("predict", help="Run prediction with a saved model")
    pred_p.add_argument("--predictor", choices=list(PREDICTORS), default="dt")
    pred_p.add_argument("--seq-length", type=int, default=DEFAULT_SEQ_LENGTH)
    pred_p.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS)
    pred_p.add_argument("--load", type=str, required=True, help="Path to saved model")
    pred_p.set_defaults(func=cmd_predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
