import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from get_features import read_features_files


MODULE_DIR = Path(__file__).resolve().parent
DATA_SET = "LDA_data1"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Screen one balanced set of unlabeled candidate pairs."
    )
    parser.add_argument("--negative_file", required=True)
    parser.add_argument("--selected_file", required=True)
    parser.add_argument("--scores_file", required=True)
    parser.add_argument("--random_seed", type=int, default=1)
    return parser.parse_args()


def make_result_writer(selected_file, scores_file):
    selected_file.parent.mkdir(parents=True, exist_ok=True)
    scores_file.parent.mkdir(parents=True, exist_ok=True)

    def writer(_pos_rna, _pos_disease, retain_rna, retain_disease, scores,
               _output_file):
        retained = np.asarray(
            [[int(rna), int(disease)]
             for rna, disease in zip(retain_rna, retain_disease)],
            dtype=int,
        )
        if retained.size == 0:
            retained = np.empty((0, 2), dtype=int)
            selected_file.write_text("", encoding="utf-8")
        else:
            np.savetxt(str(selected_file), retained, fmt="%d", delimiter=" ")

        flat_scores = np.asarray(scores, dtype=float).reshape(-1)
        if len(flat_scores) != len(retained):
            raise RuntimeError(
                "The retained-pair and score counts differ: {} versus {}."
                .format(len(retained), len(flat_scores))
            )
        np.savetxt(str(scores_file), flat_scores, fmt="%.10f")

    return writer


def main():
    args = parse_args()
    import PG1 as pg_module

    negative_file = Path(args.negative_file).resolve()
    selected_file = Path(args.selected_file).resolve()
    scores_file = Path(args.scores_file).resolve()
    if not negative_file.is_file():
        raise FileNotFoundError(negative_file)


    os.chdir(str(MODULE_DIR))
    (MODULE_DIR / "result" / DATA_SET).mkdir(parents=True, exist_ok=True)

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

    positive_features, positive_rna, positive_disease = read_features_files(
        str(MODULE_DIR / "Pos_edge.txt")
    )
    negative_features, negative_rna, negative_disease = read_features_files(
        str(negative_file)
    )
    if len(positive_features) != len(negative_features):
        raise ValueError(
            "A candidate subset must be class-balanced: {} positive pairs and "
            "{} candidate pairs were supplied."
            .format(len(positive_features), len(negative_features))
        )

    features = positive_features + negative_features
    labels = [1] * len(positive_features) + [0] * len(negative_features)
    rna_ids = positive_rna + negative_rna
    disease_ids = positive_disease + negative_disease
    print("Input labels:\n{}".format(pd.Series(labels).value_counts()))


    pg_module.generate_new_dataset = make_result_writer(
        selected_file, scores_file
    )
    agent = pg_module.PG1(
        rna_ids, disease_ids, features, labels, 2, DATA_SET
    )
    agent.train_policy()

    if not selected_file.exists() or not scores_file.exists():
        raise RuntimeError("RL screening finished without producing outputs.")


if __name__ == "__main__":
    main()
