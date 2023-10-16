import torch
from transformers import AutoTokenizer

from examples.wiki.pipeline import get_loaders


def main(model_id: int = 0):
    scores = torch.load(f"../files/results/{model_id}/wiki_if.pt", map_location="cpu")

    _, eval_train_loader, valid_loader = get_loaders(
        train_indices=None,
        valid_indices=list(range(32)),
        eval_batch_size=8,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "gpt2", use_fast=True, trust_remote_code=True
    )

    for i in range(16):
        print("=" * 80)
        print(f"{i}th data point")
        print("Sequence:")
        print(tokenizer.decode(valid_loader.dataset[i]["input_ids"]))

        print("Most influential data point")
        rank = torch.argsort(scores[i], descending=True)
        for j in range(1):
            print(f"Rank {j} (score = {scores[i][rank[j]]})")
            print(
                tokenizer.decode(eval_train_loader.dataset[int(rank[j])]["input_ids"])
            )


if __name__ == "__main__":
    main()
