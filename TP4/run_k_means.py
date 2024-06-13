import utils
from k_means import k_means
import matplotlib.pyplot as plt
#from get_bert_embedding import get_sentence_embedding

# Preguntas!!!!
# Como actualizamos el centroide si no le corresponde ningun elemento (div por 0!!!)
# La varianza nos da valores muy grandes, del orden de 10^20
# Tema de los embeddings de bert
# Como hacemos analisis del error ahora con las variaciones?

#Ver que hacemos con las columnas que son strings:
# No las tenemos en cuenta
# Las tenemos en cuenta y usamos distancia == 0 si hay igualdad y 1 si no
# Usamos embeddings de bert

#Todas las columnas son numericas menos:
#genres, imdb_id, origina_title, overview que son object pero deberian ser strings
#release_date que es object y deberia ser date

#Usar aprox max_epochs = 500 * n siendo n la dimension de los vectores de entrada

def main():
    points, df = utils.get_data()
    print_k_means(points)

    subset, subset_df = utils.get_subset()

def print_k_means(points):
    vars = []
    for k in range(1, 11):
        variation, classes, centroids, intermediate_vars = k_means(k, points, False)
        vars.append(variation)
    
    print(vars)

    plt.plot(list(range(1, 11)), vars)
    plt.title('Variation over Parition Number')
    plt.xlabel('Partition Number')
    plt.ylabel('Variation')
    plt.show()
    plt.savefig('Graphs/var_over_k.png')


if __name__ == "__main__":
    main()