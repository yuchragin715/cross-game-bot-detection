import matplotlib.pyplot as plt


def plot_trajectory(mouse_df, title=None, figsize=(7, 5)):
    x = mouse_df["dx"].cumsum()
    y = mouse_df["dy"].cumsum()

    plt.figure(figsize=figsize)
    plt.plot(x, y, linewidth=0.5)
    # plt.axis("equal")
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.show()
