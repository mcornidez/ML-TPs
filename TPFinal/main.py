import keras
import numpy as np
import matplotlib.pyplot as plt
import openTSNE
from sklearn.manifold import TSNE, trustworthiness
from sklearn.utils import shuffle
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans

sample_size = 2000

(x_train, y_train), (_, __) = keras.datasets.mnist.load_data()


reps = 2
max_perp = 15
perps_step = 5
tot_perps = int(max_perp/perps_step)

trusts = np.zeros(tot_perps)
aris = np.zeros(tot_perps)
nmis = np.zeros(tot_perps)
kl_divs = np.zeros(tot_perps)

for j in range(reps):

    x_train, y_train = shuffle(x_train, y_train)
    x_reduced = x_train[:sample_size]
    x_sample = x_reduced[:sample_size].reshape((x_reduced.shape[0], x_reduced.shape[1] * x_reduced.shape[2]))
    y_sample = y_train[:sample_size]

    for i in range(perps_step,max_perp+1,perps_step):

        step = int(i/perps_step) - 1

        model = TSNE(n_components=2, n_iter = 3000, perplexity=i)
        x_embedd = model.fit_transform(x_sample) 

        kmeans_sample = KMeans(n_clusters=10, random_state=42).fit_predict(x_sample)
        kmeans_embedd = KMeans(n_clusters=10, random_state=42).fit_predict(x_embedd)

        trusts[step] += trustworthiness(x_sample, x_embedd, n_neighbors=5)
        aris[step] += adjusted_rand_score(kmeans_sample, kmeans_embedd)
        nmis[step] += normalized_mutual_info_score(kmeans_sample, kmeans_embedd)
        kl_divs[step] += model.kl_divergence_

trusts /= reps; 
aris /= reps; 
nmis /= reps; 
kl_divs /= reps; 
    

fig, ax = plt.subplots()

# Plot each list as a curve
ax.plot(trusts, label='trusts')
ax.plot(aris, label='aris')
ax.plot(nmis, label='nmis')
ax.plot(kl_divs, label='kl_divs')

# Add labels and title
ax.set_xlabel('Perplexity')
ax.set_title('Perplexity Evaluation')

# Add a legend
ax.legend()

# Show the plot
plt.show()

def plot(x_embedd, y_sample):

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(x_embedd[:, 0], x_embedd[:, 1], c=y_sample, cmap='tab10')
    legend1 = plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title('Representacion 2D de MNIST')
    #plt.xlabel('t-SNE feature 1')
    #plt.ylabel('t-SNE feature 2')
    plt.show()

plot(x_embedd, y_sample)