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

x_train, y_train = shuffle(x_train, y_train)

x_train = x_train[:sample_size]

x_sample = x_train[:sample_size].reshape((x_train.shape[0], x_train.shape[1] * x_train.shape[2]))
y_sample = y_train[:sample_size]


model = TSNE(n_components=2, n_iter = 3000)
x_embedd = model.fit_transform(x_sample) 

kmeans_sample = KMeans(n_clusters=10, random_state=42).fit_predict(x_sample)
kmeans_embedd = KMeans(n_clusters=10, random_state=42).fit_predict(x_embedd)

trust = trustworthiness(x_sample, x_embedd, n_neighbors=5)
ari = adjusted_rand_score(kmeans_sample, kmeans_embedd)
nmi = normalized_mutual_info_score(kmeans_sample, kmeans_embedd)
kl_divergence = model.kl_divergence_
print(f'KL Divergence: {kl_divergence:.4f}')

print(f"Adjusted Rand Index (ARI): {ari}")
print(f"Normalized Mutual Information (NMI): {nmi}")
print(f"Trustworthiness: {trust}")


plt.figure(figsize=(10, 7))
scatter = plt.scatter(x_embedd[:, 0], x_embedd[:, 1], c=y_sample, cmap='tab10')
legend1 = plt.legend(*scatter.legend_elements(), title="Classes")
plt.title('t-SNE visualization of MNIST')
plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')
plt.show()
