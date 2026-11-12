import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
CONFIGS = {
    "Dataset1": {
        "directory": ROOT / "data_and_code_for_Dataset1",
        "seeds": (0, 1, 2, 3, 4, 5),
    },
    "Dataset2": {
        "directory": ROOT / "data_and_code_for_Dataset2",
        "seeds": (1233, 1234, 1235, 1236, 1237, 1238),
    },
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run all six data-split seeds using the unchanged train_LDA.py."
    )
    parser.add_argument(
        "--dataset",
        choices=("Dataset1", "Dataset2", "all"),
        required=True,
        help="Dataset to run; 'all' runs both six-seed sets sequentially.",
    )
    parser.add_argument(
        "--cuda_device",
        type=int,
        help="Physical CUDA device exposed to the unchanged training script.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun seeds whose complete archived outputs already exist.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the exact six-seed task list without starting training.",
    )
    return parser.parse_args()


def expected_archived_files(seed_directory):
    paths = [
        seed_directory / "inter_holdout_positive.txt",
        seed_directory / "inter_holdout_negative.txt",
        seed_directory / "training.log",
        seed_directory / "run_metadata.json",
    ]
    for fold in range(1, 6):
        paths.extend(
            [
                seed_directory / ("sample{}train.txt".format(fold)),
                seed_directory / ("sample{}valid.txt".format(fold)),
                seed_directory / ("label{}train.txt".format(fold)),
                seed_directory / ("label{}valid.txt".format(fold)),
                seed_directory / ("{}true_valid.txt".format(fold)),
                seed_directory / ("{}predict_valid.txt".format(fold)),
                seed_directory / ("{}true_ind_test.txt".format(fold)),
                seed_directory / ("{}predict_ind_test.txt".format(fold)),
                seed_directory / "saved_models" / ("best_model_fold_{}.pt".format(fold - 1)),
            ]
        )
    return paths


def is_complete(dataset_name, seed):
    seed_directory = (
        CONFIGS[dataset_name]["directory"]
        / "six_seed_results"
        / ("seed_{}".format(seed))
    )
    metadata_path = seed_directory / "run_metadata.json"
    if not metadata_path.is_file():
        return False
    try:
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, ValueError):
        return False
    return (
        metadata.get("dataset") == dataset_name
        and int(metadata.get("data_split_seed", -1)) == int(seed)
        and all(path.is_file() for path in expected_archived_files(seed_directory))
    )


def run_with_log(command, working_directory, environment, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [str(item) for item in command],
            cwd=str(working_directory),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        if process.stdout is None:
            raise RuntimeError("Could not capture train_LDA.py output.")
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def archive_text_outputs(module_directory, seed_directory):
    filenames = ["inter_holdout_positive.txt", "inter_holdout_negative.txt"]
    for fold in range(1, 6):
        filenames.extend(
            [
                "sample{}train.txt".format(fold),
                "sample{}valid.txt".format(fold),
                "label{}train.txt".format(fold),
                "label{}valid.txt".format(fold),
                "{}true_valid.txt".format(fold),
                "{}predict_valid.txt".format(fold),
                "{}true_ind_test.txt".format(fold),
                "{}predict_ind_test.txt".format(fold),
            ]
        )
    for filename in filenames:
        source = module_directory / filename
        if not source.is_file():
            raise FileNotFoundError(
                "The unchanged training run did not produce expected file: {}".format(source)
            )
        shutil.copy2(str(source), str(seed_directory / filename))


def run_one(dataset_name, seed, options, environment):
    config = CONFIGS[dataset_name]
    module_directory = config["directory"]
    seed_directory = module_directory / "six_seed_results" / ("seed_{}".format(seed))
    if is_complete(dataset_name, seed) and not options.force:
        print("Skipping completed {} seed {}.".format(dataset_name, seed))
        return

    seed_directory.mkdir(parents=True, exist_ok=True)
    model_directory = seed_directory / "saved_models"
    command = [
        sys.executable,
        "train_LDA.py",
        "--random_seed",
        str(seed),
        "--model_dir",
        str(model_directory),
    ]
    print("\n[{} seed {}] {}".format(
        dataset_name, seed, subprocess.list2cmdline([str(item) for item in command])
    ))
    run_with_log(
        command,
        module_directory,
        environment,
        seed_directory / "training.log",
    )
    archive_text_outputs(module_directory, seed_directory)
    metadata = {
        "model": "PRMHGRL",
        "dataset": dataset_name,
        "data_split_seed": int(seed),
        "formal_dataset_seeds": list(config["seeds"]),
        "fold_count": 5,
        "internal_holdout": "fixed across data-split seeds",
        "model_initialization": "unchanged historical fixed seed in model.py",
        "training_script": "train_LDA.py",
        "command": subprocess.list2cmdline([str(item) for item in command]),
    }
    with (seed_directory / "run_metadata.json").open("w", encoding="utf-8") as output:
        json.dump(metadata, output, indent=2)
    print("Archived {} seed {} to {}".format(dataset_name, seed, seed_directory))


def main():
    options = parse_arguments()
    datasets = (
        ("Dataset1", "Dataset2") if options.dataset == "all" else (options.dataset,)
    )
    tasks = [
        (dataset_name, seed)
        for dataset_name in datasets
        for seed in CONFIGS[dataset_name]["seeds"]
    ]
    print("Six-seed experiment tasks ({}):".format(len(tasks)))
    for dataset_name, seed in tasks:
        print("  {} / seed {}".format(dataset_name, seed))
    if options.dry_run:
        return

    environment = os.environ.copy()
    if options.cuda_device is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(options.cuda_device)
        print("CUDA physical device: {}".format(options.cuda_device))
    for dataset_name, seed in tasks:
        run_one(dataset_name, seed, options, environment)


if __name__ == "__main__":
    main()
