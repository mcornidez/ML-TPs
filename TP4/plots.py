import matplotlib.pyplot as plt

import os
os.makedirs("./Out", exist_ok=True)

def heatmap(matrix, file, title=None, text=None):
    plt.figure()

    if text is None:
        text = []
    plt.imshow(matrix, cmap="RdYlGn")
    plt.colorbar()

    # Annotate the heatmap with the values
    for i in range(len(text)):
        for j in range(len(text[0])):
            plt.text(
                j, i, text[i][j], ha="center", va="center", color="black", fontsize=9
            )

    if title is not None:
        plt.title(title)

    plt.savefig(f"./Out/{file}")
