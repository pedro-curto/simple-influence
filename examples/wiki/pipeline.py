from itertools import chain
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
)

GLUE_TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


class LanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = AutoConfig.from_pretrained(
            "gpt2",
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            "gpt2",
            from_tf=False,
            config=self.config,
            ignore_mismatched_sizes=False,
            trust_remote_code=True,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits


def construct_model() -> nn.Module:
    return LanguageModel()


def get_loaders(
    eval_batch_size: int = 32,
    train_indices: Optional[List[int]] = None,
    valid_indices: Optional[List[int]] = None,
) -> Tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    train_batch_size = 16

    train_loader = get_wiki_dataloader(
        batch_size=train_batch_size,
        split="train",
        indices=train_indices,
    )
    eval_train_loader = get_wiki_dataloader(
        batch_size=eval_batch_size,
        split="eval_train",
        indices=train_indices,
    )
    valid_loader = get_wiki_dataloader(
        batch_size=eval_batch_size,
        split="valid",
        indices=valid_indices,
    )
    return train_loader, eval_train_loader, valid_loader


def get_wiki_dataloader(
    batch_size: int = 32,
    split: str = "train",
    indices: List[int] = None,
) -> torch.utils.data.DataLoader:
    raw_datasets = load_dataset("wikitext", "wikitext-2-raw-v1")

    tokenizer = AutoTokenizer.from_pretrained(
        "gpt2", use_fast=True, trust_remote_code=True
    )

    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=None,
        remove_columns=column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )
    block_size = 512

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=None,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    if split == "train" or split == "eval_train":
        train_dataset = lm_datasets["train"]
        ds = train_dataset
    else:
        eval_dataset = lm_datasets["validation"]
        ds = eval_dataset

    if indices is not None:
        ds = ds.select(indices)

    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=split == "train",
        collate_fn=default_data_collator,
    )
