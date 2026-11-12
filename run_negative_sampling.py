import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the complete RL negative-candidate screening workflow."
    )
    parser.add_argument("--dataset", choices=("Dataset1", "Dataset2"), required=True)
    parser.add_argument("--random_seed", type=int, default=1,
                        help="Historical random seed used by the RL selector.")
    parser.add_argument("--final_selection_seed", type=int, default=1,
                        help="Seed for sampling the final balanced set.")
    parser.add_argument("--cuda_device", type=int)
    parser.add_argument("--subset_start", type=int, default=1,
                        help="First balanced subset to process (1-based).")
    parser.add_argument("--subset_end", type=int,
                        help="Last balanced subset to process (inclusive).")
    parser.add_argument("--output_root", default="negative_sampling_results")
    parser.add_argument("--force", action="store_true",
                        help="Recompute subset outputs that already exist.")
    parser.add_argument("--dry_run", action="store_true",
                        help="Validate inputs and show the planned command only.")
    return parser.parse_args()


def load_pairs(path):
    path = Path(path)
    pairs = np.atleast_2d(np.loadtxt(str(path), dtype=int))
    if pairs.shape[1] != 2:
        raise ValueError("{} must contain exactly two columns.".format(path))
    return pairs


def load_optional_pairs(path):
    path = Path(path)
    if not path.exists() or not path.read_text(encoding="utf-8").strip():
        return np.empty((0, 2), dtype=int)
    return load_pairs(path)


def write_pairs(path, pairs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(str(path), np.asarray(pairs, dtype=int), fmt="%d", delimiter=" ")


def run_logged(command, cwd, environment, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def main():
    args = parse_args()
    dataset_dir = ROOT / args.dataset
    positive_file = dataset_dir / "Pos_edge.txt"
    candidate_file = dataset_dir / "Neg_edges" / "Neg_edge.txt"
    worker_file = dataset_dir / "main.py"
    for required in (positive_file, candidate_file, worker_file):
        if not required.is_file():
            raise FileNotFoundError(required)

    positive_pairs = load_pairs(positive_file)
    candidate_pairs = load_pairs(candidate_file)
    subset_size = len(positive_pairs)
    complete_subset_count = len(candidate_pairs) // subset_size
    if complete_subset_count < 1:
        raise ValueError(
            "The candidate pool is smaller than the known-positive set and "
            "cannot form a balanced subset."
        )

    subset_end = args.subset_end or complete_subset_count
    if not 1 <= args.subset_start <= subset_end <= complete_subset_count:
        raise ValueError(
            "The requested subset range is outside the complete balanced "
            "candidate subsets available from the input pool."
        )

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_dir = output_root / args.dataset
    subset_dir = output_dir / "subsets"
    log_dir = output_dir / "logs"

    environment = os.environ.copy()
    environment.setdefault("MPLBACKEND", "Agg")
    if args.cuda_device is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)

    print("Dataset: {}".format(args.dataset))
    print("Known positive pairs: {}".format(len(positive_pairs)))
    print("Unlabeled candidate pairs: {}".format(len(candidate_pairs)))
    print(
        "Processing complete class-balanced candidate subsets in the requested "
        "range."
    )
    if args.dry_run:
        print(
            "Dry run successful. Outputs would be written to: {}"
            .format(output_dir)
        )
        return

    subset_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    for subset_number in range(args.subset_start, subset_end + 1):
        start = (subset_number - 1) * subset_size
        stop = subset_number * subset_size
        prefix = "subset_{:03d}".format(subset_number)
        input_file = subset_dir / (prefix + "_input.txt")
        selected_file = subset_dir / (prefix + "_selected.txt")
        scores_file = subset_dir / (prefix + "_scores.txt")
        log_file = log_dir / (prefix + ".log")

        if selected_file.exists() and scores_file.exists() and not args.force:
            print("Skipping completed candidate subset {}.".format(subset_number))
            continue

        write_pairs(input_file, candidate_pairs[start:stop])
        command = [
            sys.executable,
            str(worker_file),
            "--negative_file", str(input_file),
            "--selected_file", str(selected_file),
            "--scores_file", str(scores_file),
            "--random_seed", str(args.random_seed),
        ]
        print("\n> {}".format(subprocess.list2cmdline(command)))
        run_logged(command, dataset_dir, environment, log_file)


    all_selected_files = [
        subset_dir / "subset_{:03d}_selected.txt".format(index)
        for index in range(1, complete_subset_count + 1)
    ]
    all_score_files = [
        subset_dir / "subset_{:03d}_scores.txt".format(index)
        for index in range(1, complete_subset_count + 1)
    ]
    missing = [
        path for path in all_selected_files + all_score_files
        if not path.exists()
    ]
    if missing:
        print(
            "Requested subset range completed. The final set will be built "
            "after all balanced subsets have been processed."
        )
        return

    retained_parts = []
    score_parts = []
    for selected_file, scores_file in zip(all_selected_files, all_score_files):
        selected = load_optional_pairs(selected_file)
        scores = np.atleast_1d(np.loadtxt(str(scores_file), dtype=float))
        if selected.shape[0] != scores.shape[0]:
            raise RuntimeError(
                "Pair/score count mismatch for {}.".format(selected_file)
            )
        retained_parts.append(selected)
        score_parts.append(scores)

    retained_pairs = np.vstack(retained_parts)
    retained_scores = np.concatenate(score_parts)
    retained_as_tuples = [tuple(pair) for pair in retained_pairs.tolist()]
    if len(set(retained_as_tuples)) != len(retained_as_tuples):
        raise RuntimeError("Duplicate pairs occurred across disjoint subsets.")
    if len(retained_pairs) < subset_size:
        raise RuntimeError(
            "The selector retained {} candidates, fewer than the {} required "
            "for the final balanced set."
            .format(len(retained_pairs), subset_size)
        )

    positive_set = set(tuple(pair) for pair in positive_pairs.tolist())
    if positive_set.intersection(retained_as_tuples):
        raise RuntimeError("Retained candidates overlap known positive pairs.")

    selector = random.Random(args.final_selection_seed)
    final_indices = selector.sample(range(len(retained_pairs)), subset_size)
    final_pairs = retained_pairs[final_indices]
    final_scores = retained_scores[final_indices]

    write_pairs(output_dir / "selected_candidate_pool.txt", retained_pairs)
    np.savetxt(
        str(output_dir / "selected_candidate_scores.txt"),
        retained_scores,
        fmt="%.10f",
    )
    write_pairs(output_dir / "final_negative.txt", final_pairs)
    np.savetxt(
        str(output_dir / "final_negative_scores.txt"),
        final_scores,
        fmt="%.10f",
    )

    manifest = {
        "dataset": args.dataset,
        "random_seed": args.random_seed,
        "final_selection_seed": args.final_selection_seed,
        "known_positive_count": int(len(positive_pairs)),
        "unlabeled_candidate_count": int(len(candidate_pairs)),
        "processed_candidate_count": int(complete_subset_count * subset_size),
        "unused_incomplete_remainder_count": int(
            len(candidate_pairs) - complete_subset_count * subset_size
        ),
        "retained_candidate_count": int(len(retained_pairs)),
        "final_negative_count": int(len(final_pairs)),
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)

    print("\nFinal negative set: {}".format(output_dir / "final_negative.txt"))


if __name__ == "__main__":
    main()
