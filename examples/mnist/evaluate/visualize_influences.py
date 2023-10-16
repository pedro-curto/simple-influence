import matplotlib.pyplot as plt
import numpy as np
import torch

from examples.mnist.pipeline import get_loaders


def main(data_name: str = "mnist", model_id: int = 0):
    scores = torch.load(
        f"../files/results/{model_id}/{data_name}_if.pt", map_location="cpu"
    )

    _, eval_train_loader, valid_loader = get_loaders(
        data_name=data_name,
        train_indices=None,
        valid_indices=list(range(6)),
        eval_batch_size=8,
    )

    for i, idx in enumerate(range(6)):
        fig, axs = plt.subplots(ncols=7, figsize=(15, 3))
        fig.suptitle("Top Influential Images")
        axs[0].imshow(
            np.transpose(
                np.reshape(valid_loader.dataset[idx][0], (1, 28, 28)), (1, 2, 0)
            )
        )
        axs[0].axis("off")
        axs[0].set_title("Target Image")
        top_idxs = np.argsort(scores[idx].numpy())[::-1][:5]
        for ii, train_im_ind in enumerate(top_idxs):
            axs[ii + 2].imshow(
                np.transpose(
                    np.reshape(eval_train_loader.dataset[train_im_ind][0], (1, 28, 28)),
                    (1, 2, 0),
                )
            )
            axs[ii + 2].axis("off")
        axs[1].set_axis_off()
        fig.show()


if __name__ == "__main__":
    main()
