import torch
from tqdm import tqdm, trange
from transformers import AutoTokenizer

from examples.wiki.pipeline import get_loaders


def find_largest_ngram_overlap(str1, str2):
    len_str1, len_str2 = len(str1), len(str2)
    table = [[0] * (len_str2 + 1) for _ in range(len_str1 + 1)]
    longest = 0
    l_end = 0

    for i in range(1, len_str1 + 1):
        for j in range(1, len_str2 + 1):
            if str1[i - 1] == str2[j - 1]:
                table[i][j] = table[i - 1][j - 1] + 1
                if table[i][j] > longest:
                    longest = table[i][j]
                    l_end = i
            else:
                table[i][j] = 0

    if longest == 0:
        return ""
    return str1[l_end - longest : l_end]


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
        querry_sequence = tokenizer.decode(valid_loader.dataset[i]["input_ids"])
        print(querry_sequence)

        print("Most influential data point")
        rank = torch.argsort(scores[i], descending=True)
        for j in range(3):
            influential_sequence = tokenizer.decode(
                eval_train_loader.dataset[int(rank[j])]["input_ids"]
            )
            overlap = find_largest_ngram_overlap(querry_sequence, influential_sequence)
            print("-" * 80)
            print(f"Rank {j} (score = {scores[i][rank[j]]})")
            print(f"Overlap: {overlap}")
            print(f"Influential Sequence: {influential_sequence}")


if __name__ == "__main__":
    main()
