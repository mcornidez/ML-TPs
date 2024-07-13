import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, trustworthiness
from sklearn.utils import shuffle
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
import os

os.makedirs("./Out", exist_ok=True)


def perplexity_test():
    sample_size = 2000

    (x_train, y_train), (_, __) = keras.datasets.mnist.load_data()

    reps = 2
    max_perp = 15
    perps_step = 5
    tot_perps = int(max_perp / perps_step)

    trusts = np.zeros(tot_perps)
    aris = np.zeros(tot_perps)
    nmis = np.zeros(tot_perps)
    kl_divs = np.zeros(tot_perps)

    for j in range(reps):

        x_train, y_train = shuffle(x_train, y_train)  # type: ignore
        x_reduced = np.array(x_train[:sample_size])
        x_sample = x_reduced[:sample_size].reshape(
            (x_reduced.shape[0], x_reduced.shape[1] * x_reduced.shape[2])
        )
        y_sample = np.array(y_train[:sample_size])

        for i in range(perps_step, max_perp + 1, perps_step):

            step = int(i / perps_step) - 1

            model = TSNE(n_components=2, n_iter=3000, perplexity=i)
            x_embedd = model.fit_transform(x_sample)

            kmeans_sample = KMeans(n_clusters=10, random_state=42).fit_predict(x_sample)
            kmeans_embedd = KMeans(n_clusters=10, random_state=42).fit_predict(x_embedd)

            trusts[step] += trustworthiness(x_sample, x_embedd, n_neighbors=5)
            aris[step] += adjusted_rand_score(kmeans_sample, kmeans_embedd)
            nmis[step] += normalized_mutual_info_score(kmeans_sample, kmeans_embedd)
            kl_divs[step] += model.kl_divergence_

    trusts /= reps
    aris /= reps
    nmis /= reps
    kl_divs /= reps

    fig, ax = plt.subplots()

    # Plot each list as a curve
    ax.plot(trusts, label="trusts")
    ax.plot(aris, label="aris")
    ax.plot(nmis, label="nmis")
    ax.plot(kl_divs, label="kl_divs")

    # Add labels and title
    ax.set_xlabel("Perplexity")
    ax.set_title("Perplexity Evaluation")

    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()


def metric_test():
    sample_size = 2000
    perplexity = 15
    (x_train, y_train), (_, __) = keras.datasets.mnist.load_data()

    reps = 5

    metrics = ["euclidean", "cosine", "manhattan"]

    l = len(metrics)

    trusts = np.zeros(l)
    aris = np.zeros(l)
    nmis = np.zeros(l)
    kl_divs = np.zeros(l)

    for rep in range(reps):
        print(f"rep {rep}")

        x_train, y_train = shuffle(x_train, y_train)  # type: ignore
        x_reduced = np.array(x_train[:sample_size])
        x_sample = x_reduced[:sample_size].reshape(
            (x_reduced.shape[0], x_reduced.shape[1] * x_reduced.shape[2])
        )
        y_sample = np.array(y_train[:sample_size])

        for i, metric in enumerate(metrics):

            print(f"metric {metric}")

            model = TSNE(n_iter=3000, perplexity=perplexity, metric=metric)
            x_embedd = model.fit_transform(x_sample)

            kmeans_sample = KMeans(n_clusters=10, random_state=42).fit_predict(x_sample)
            kmeans_embedd = KMeans(n_clusters=10, random_state=42).fit_predict(x_embedd)

            trusts[i] += trustworthiness(x_sample, x_embedd, n_neighbors=5)
            aris[i] += adjusted_rand_score(kmeans_sample, kmeans_embedd)
            nmis[i] += normalized_mutual_info_score(kmeans_sample, kmeans_embedd)
            kl_divs[i] += model.kl_divergence_

            if rep == reps - 1:
                plot(x_embedd, y_sample, metric)

    trusts /= reps
    aris /= reps
    nmis /= reps
    kl_divs /= reps

    _, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].bar(metrics, trusts)
    axs[0, 0].set_title("Trustworthiness")
    axs[0, 0].set_ylabel("Trustworthiness")
    axs[0, 0].set_ylim(0.9, 1.0)

    axs[0, 1].bar(metrics, aris)
    axs[0, 1].set_title("Adjusted Rand Score")
    axs[0, 1].set_ylabel("Adjusted Rand Score")
    axs[0, 1].set_ylim(0.4, 0.5)

    axs[1, 0].bar(metrics, nmis)
    axs[1, 0].set_title("Normalized Mutual Information")
    axs[1, 0].set_ylabel("Normalized Mutual Information")
    axs[1, 0].set_ylim(0.55, 0.65)

    axs[1, 1].bar(metrics, kl_divs)
    axs[1, 1].set_title("KL Divergence")
    axs[1, 1].set_ylabel("KL Divergence")
    axs[1, 1].set_ylim(1.2, 1.4)

    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    plt.savefig("./Out/metrics.png")

def init_test():
    sample_size = 2000
    perplexity = 15
    (x_train, y_train), (_, __) = keras.datasets.mnist.load_data()

    reps = 5

    inits = ["random", "pca"]

    l = len(inits)

    trusts = np.zeros(l)
    aris = np.zeros(l)
    nmis = np.zeros(l)
    kl_divs = np.zeros(l)

    for rep in range(reps):
        print(f"rep {rep}")

        x_train, y_train = shuffle(x_train, y_train)  # type: ignore
        x_reduced = np.array(x_train[:sample_size])
        x_sample = x_reduced[:sample_size].reshape(
            (x_reduced.shape[0], x_reduced.shape[1] * x_reduced.shape[2])
        )
        y_sample = np.array(y_train[:sample_size])

        for i, init in enumerate(inits):

            print(f"init {init}")

            model = TSNE(n_iter=3000, perplexity=perplexity, init=init)
            x_embedd = model.fit_transform(x_sample)

            kmeans_sample = KMeans(n_clusters=10, random_state=42).fit_predict(x_sample)
            kmeans_embedd = KMeans(n_clusters=10, random_state=42).fit_predict(x_embedd)

            trusts[i] += trustworthiness(x_sample, x_embedd, n_neighbors=5)
            aris[i] += adjusted_rand_score(kmeans_sample, kmeans_embedd)
            nmis[i] += normalized_mutual_info_score(kmeans_sample, kmeans_embedd)
            kl_divs[i] += model.kl_divergence_

            if rep == reps - 1:
                plot(x_embedd, y_sample, init)

    trusts /= reps
    aris /= reps
    nmis /= reps
    kl_divs /= reps

    _, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].bar(inits, trusts)
    axs[0, 0].set_title("Trustworthiness")
    axs[0, 0].set_ylabel("Trustworthiness")
    axs[0, 0].set_ylim(0.9, 1.0)

    axs[0, 1].bar(inits, aris)
    axs[0, 1].set_title("Adjusted Rand Score")
    axs[0, 1].set_ylabel("Adjusted Rand Score")
    axs[0, 1].set_ylim(0.4, 0.5)

    axs[1, 0].bar(inits, nmis)
    axs[1, 0].set_title("Normalized Mutual Information")
    axs[1, 0].set_ylabel("Normalized Mutual Information")
    axs[1, 0].set_ylim(0.55, 0.65)

    axs[1, 1].bar(inits, kl_divs)
    axs[1, 1].set_title("KL Divergence")
    axs[1, 1].set_ylabel("KL Divergence")
    axs[1, 1].set_ylim(1.2, 1.4)

    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    plt.savefig("./Out/init.png")

def perplexity_clusters():
    sample_size = 2000
    (x_train, y_train), (_, __) = keras.datasets.mnist.load_data()

    perplexities = [20, 40, 60, 80, 100]

    x_train, y_train = shuffle(x_train, y_train)  # type: ignore
    x_reduced = np.array(x_train[:sample_size])
    x_sample = x_reduced[:sample_size].reshape(
        (x_reduced.shape[0], x_reduced.shape[1] * x_reduced.shape[2])
    )
    y_sample = np.array(y_train[:sample_size])

    for p in perplexities:

        print(f"perplexity {p}")

        model = TSNE(n_iter=3000, perplexity=p)
        x_embedd = model.fit_transform(x_sample)

        plot(x_embedd, y_sample, f"Perplexity = {p}")

def plot(x_embedd, y_sample, title="Representacion 2D de MNIST"):
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(x_embedd[:, 0], x_embedd[:, 1], c=y_sample, cmap="tab10")
    legend1 = plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title(title)
    # plt.xlabel('t-SNE feature 1')
    # plt.ylabel('t-SNE feature 2')
    plt.savefig(f"./Out/cluster_{title}.png")


perplexity_clusters()
