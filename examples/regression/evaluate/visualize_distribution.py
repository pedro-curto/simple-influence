import matplotlib.pyplot as plt
import torch


def main(data_name: str = "concrete", model_id: int = 0):
    scores = torch.load(
        f"../files/results/{model_id}/{data_name}_if.pt", map_location="cpu"
    )

    eval_idxs = [0, 1, 2]
    for idx in eval_idxs:
        plt.plot(torch.sort(scores[idx]).values)
        plt.title(f"Influence Distribution for Validation Index {idx}")
        plt.ylabel("Influence Scores")
        plt.xlabel("Training Data Example")
        plt.grid()
        plt.show()


if __name__ == "__main__":
    main()
