from conditionalModel import ConditionalModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():

    df = pd.read_csv("Data/binary.csv")
    data = df[["admit", "gre", "gpa", "rank"]]
    data["gre"] = data["gre"].apply(lambda x: 1 if x >= 500 else 0)
    data["gpa"] = data["gpa"].apply(lambda x: 1 if x >= 3 else 0)

    ranks = list(data["rank"])
    rank_counts = np.zeros((4,1))
    for i in range(4):
        rank_counts[i][0] = ranks.count(i+1)
    total_count = len(ranks)
    rank_probs = rank_counts/total_count
    print("Ranks probs:", rank_probs)

    rank = 1
    admit_conditional = ConditionalModel(
        data[["admit"]].to_numpy(), data[["gre", "gpa", "rank"]].to_numpy()
        #np.where(data[["admit"]].to_numpy() == 0, 1, 0), data[["gre", "gpa", "rank"]].to_numpy()
    )
    classes_admit, probs_admit = admit_conditional.calculate_conditional()
    print(classes_admit)
    print("Admit cond probs:", probs_admit)

    gpa_conditional = ConditionalModel(
        data[["gpa"]].to_numpy(), data[["rank"]].to_numpy()
    )
    classes_gpa, probs_gpa = gpa_conditional.calculate_conditional()
    print("Gpa cond probs:", probs_gpa)

    gre_conditional = ConditionalModel(
        data[["gre"]].to_numpy(), data[["rank"]].to_numpy()
    )
    classes_gre, probs_gre = gre_conditional.calculate_conditional()
    print("Gre cond probs:", probs_gre)

    admit2_conditional = ConditionalModel(
        data[["admit"]].to_numpy(), data[["rank"]].to_numpy()
        #np.where(data[["admit"]].to_numpy() == 0, 1, 0), data[["rank"]].to_numpy()
    )
    classes_admit2, probs_admit2 = admit2_conditional.calculate_conditional()

    #Agrupamos probabilidades condicionales compuestas en base al rango
    ranks = list(map(lambda x: x[2], classes_admit))
    print(probs_admit) 
    classified_admit_probs = np.zeros((4,4))
    for i in range(4):
        classified_admit_probs[i] = probs_admit[np.where(np.array(ranks) == i+1)[0]]

    # P(gpa | rank) * P(gre | rank)
    gre_gpa_prods = np.array([(1 - probs_gre) * (1 - probs_gpa), (1 - probs_gre) * probs_gpa, probs_gre * (1 - probs_gpa), probs_gre * probs_gpa])
    print(gre_gpa_prods)
    # (P(gre | rank) * P(gpa | rank)) * P(rank) * P(admit | gre, gpa, rank)
    prods = gre_gpa_prods.transpose() * rank_probs * classified_admit_probs

    print("Probability of rejection given a student studied in a rank 1 school:")
    print(prods[0].sum()/prods.sum())

    index = np.where(classes_admit2)[rank - 1][0]
    print("Probability of rejection given a student studied in a rank 1 school using v2:")
    print(1 - probs_admit2[index])

    # Probabilidad admision para caso rango = 2, gre = 450 y gpa = 3.5
    condition = [0, 1, 2]
    classes_admit, probs_admit = admit_conditional.calculate_conditional()
    index = np.where(list(map(lambda x: np.all(x), classes_admit == condition)))[0][0]
    print(
        "Probability of admission given a student studied in a rank 2 school, got a gre of 450 and a gpa of 3.5:"
    )
    print(probs_admit[index])


if __name__ == "__main__":
    main()
