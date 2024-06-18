import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

import os

os.makedirs("./Out", exist_ok=True)


def heatmap(matrix, file, title=None, text=None, fontSize=9):
    plt.figure()

    if text is None:
        text = []
    plt.imshow(matrix, cmap="RdYlGn")
    plt.colorbar()

    # Annotate the heatmap with the values
    for i in range(len(text)):
        for j in range(len(text[0])):
            plt.text(
                j, i, text[i][j], ha="center", va="center", color="black", fontsize=fontSize
            )

    if title is not None:
        plt.title(title)

    plt.savefig(f"./Out/{file}")


def dendogram(linkage, file_name, format):
    plt.figure()

    dendrogram(linkage, leaf_rotation=90)
    plt.savefig(file_name, format=format, dpi=300)

    plt.show()
