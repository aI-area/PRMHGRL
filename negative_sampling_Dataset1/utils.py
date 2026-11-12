from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def genDataLoader(sents, labels, shuffle, batch_size):
    data_gen = TensorDataset(
        torch.FloatTensor(np.asarray(sents)),
        torch.FloatTensor(np.asarray(labels)),
    )
    return DataLoader(data_gen, shuffle=shuffle, batch_size=batch_size)


def genDataLoader_with_action(sents, labels, actions, shuffle, batch_size):
    data_gen = TensorDataset(
        torch.FloatTensor(np.asarray(sents)),
        torch.FloatTensor(np.asarray(labels)),
        torch.LongTensor(np.asarray(actions)).unsqueeze(1),
    )
    return DataLoader(data_gen, shuffle=shuffle, batch_size=batch_size)


def generate_new_dataset(pos_rna, pos_pro, retain_rna, retain_pro, scores,
                         output_file):

    output_prefix = Path(output_file)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    selected_file = Path(str(output_prefix) + "_selected_negative_samples.txt")
    scores_file = Path(str(output_prefix) + "_selection_scores.txt")
    retained = np.asarray(
        [[int(rna), int(disease)]
         for rna, disease in zip(retain_rna, retain_pro)],
        dtype=int,
    )
    if retained.size == 0:
        selected_file.write_text("", encoding="utf-8")
    else:
        np.savetxt(str(selected_file), retained, fmt="%d", delimiter=" ")
    np.savetxt(str(scores_file), np.asarray(scores).reshape(-1), fmt="%.10f")
