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
    rank_counts = np.zeros(4)
    for i in range(4):
        rank_counts[i] = ranks.count(i+1)
    total_count = len(ranks)
    rank_probs = rank_counts/total_count
    print("Ranks probs:", rank_probs)

    rank = 1
    admit_conditional = ConditionalModel(
        data[["admit"]].to_numpy(), data[["gre", "gpa", "rank"]].to_numpy()
    )
    classes_admit, probs_admit = admit_conditional.calculate_conditional()
    print(f"Classes admit: {classes_admit}")
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

    term1 = rank_probs[0]*(1 - probs_admit[0])*(1-probs_gre[0])*(1-probs_gpa[0])
    term2 = rank_probs[0]*(1 - probs_admit[4])*(1-probs_gre[0])*probs_gpa[0]
    term3 = rank_probs[0]*(1 - probs_admit[8])*probs_gre[0]*(1-probs_gpa[0])
    term4 = rank_probs[0]*(1 - probs_admit[12])*probs_gre[0]*probs_gpa[0]

    result = (term1+term2+term3+term4) / rank_probs[0]

    print("Terms for part a: ", term1, term2, term3, term4)

    print("Denominator for part a: ", rank_probs[0])

    print(f"Probability of rejection given a student studied in a rank 1 school: {result}")

    # Probabilidad admision para caso rango = 2, gre = 450 y gpa = 3.5
    print(
        f"Probability of admission given a student studied in a rank 2 school, got a gre of 450 and a gpa of 3.5: {probs_admit[5]}"
    )


if __name__ == "__main__":
    main()
