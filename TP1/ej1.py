#import naiveBayes as nb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    df = pd.read_excel("PreferenciasBritanicos.xlsx")

    prefs = df[["scones", "cerveza", "wiskey", "avena", "futbol"]].to_numpy()
    nac = df["Nacionalidad"].to_numpy()

    nac_set = set(nac)
    class_indexes = list(map(lambda x: np.where(nac == x)[0], nac_set))
    counts = np.asarray(list(map(lambda x: x.size, class_indexes)))
    total_count = counts.sum()
    class_probs = counts / total_count
    dissected_data = np.asarray(list(map(lambda x: prefs[x].transpose().sum(axis=1), class_indexes)))
    indiv_prob = dissected_data/counts.reshape((2,1))
    probs = indiv_prob.prod(axis=1) * class_probs
    final_probs = list(map(lambda x: x/probs.sum(), probs))
    print(nac_set)
    print(class_indexes)
    print(class_probs)
    print(dissected_data)
    print(indiv_prob)
    print(probs)
    print(final_probs)



    #classifier = nb.Classifier(prefs, nac)

    #print(classify())


if __name__ == "__main__":
    main()
