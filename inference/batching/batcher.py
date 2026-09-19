from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class Batch:
    input_ids: Tensor
    attention_mask: Tensor


def pad_sequences(
    sequences: list[list[int]],
    pad_token_id: int,
) -> Batch:
    if not sequences:
        raise ValueError("sequences cannot be empty")

    max_length = max(len(sequence) for sequence in sequences)

    input_ids = torch.full(
        (len(sequences), max_length),
        pad_token_id,
        dtype=torch.long,
    )

    attention_mask = torch.zeros_like(input_ids)

    for row, sequence in enumerate(sequences):
        length = len(sequence)

        if length:
            input_ids[row, :length] = torch.tensor(
                sequence,
                dtype=torch.long,
            )
            attention_mask[row, :length] = 1

    return Batch(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
