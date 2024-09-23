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
    df = df[df["kaggle_test_score"] < 30]
    for model_type, group3 in df.groupby("model_type"):
        plt.figure(figsize=(10, 5))
        for lr, group2 in group3.groupby("learning_rate"):
            print(group2)
            plt.scatter(
                group2["num_params"],
                group2["kaggle_test_score"],
                label=f"lr={lr}",
                color=color_map[lr],
                marker=marker_map_lr[lr],
                alpha=0.5,
            )
        plt.suptitle(f"Kaggle score vs number of parameters ({model_type})")
        plt.xlabel("Number of parameters")
        plt.ylabel("Kaggle test score")
        # show legend for the first plot only
        plt.legend(loc=(0.6, 0.6))
        plt.grid()
        plt.savefig(f"score_{model_type}.png", dpi=300)
