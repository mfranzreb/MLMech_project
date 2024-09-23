import pandas as pd
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt


if __name__ == "__main__":

    df = pd.read_csv("results.csv")
    # assign a color to each "learning_rate"
    colors = ["r", "b", "y", "b", "k"]
    color_map = dict(zip(pd.unique(df["learning_rate"]), colors))
    # assign a marker to each "model_type"
    markers = ["o", "x", "v", "^", "s"]
    marker_map = dict(zip(pd.unique(df["model_type"]), markers))
    marker_map_lr = dict(zip(pd.unique(df["learning_rate"]), markers))
    # remove rows with "kaggle_score" > 1000
    df = df[df["num_params"] == 3000]
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    for reg, group in df.groupby("regularization"):
        for lr, group2 in group.groupby("learning_rate"):
            axs[0].scatter(
                group2["learning_rate"],
                group2["kaggle_test_score"],
                color="r" if reg == 0.0 else "b",
                alpha=0.5,
            )
            axs[1].scatter(
                group2["learning_rate"],
                group2["training_time"],
                color="r" if reg == 0.0 else "b",
                alpha=0.5,
            )
        plt.suptitle(f"Effect of regularization")
        axs[0].set_xlabel("Learning rate")
        axs[0].set_ylabel("Kaggle test score")
        axs[1].set_xlabel("Learning rate")
        axs[1].set_ylabel("Training time (s)")
        # add legend where regularization is indicated by color if True: blue, else red
        axs[0].legend(
            [
                matplotlib.lines.Line2D(
                    [0], [0], marker="o", color="w", markerfacecolor="r", markersize=10
                ),
                matplotlib.lines.Line2D(
                    [0], [0], marker="o", color="w", markerfacecolor="b", markersize=10
                ),
            ],
            ["Regularization = False", "Regularization = True"],
        )
        plt.grid()
        plt.savefig(f"results/regularization.png", dpi=300)
