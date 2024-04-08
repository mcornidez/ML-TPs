from conditionalModel import ConditionalModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():

    df = pd.read_csv("binary.csv")
    data = df[["admit", "gre", "gpa", "rank"]]
    data["gre"] = data["gre"].apply(lambda x: 1 if x >= 500 else 0)
    data["gpa"] = data["gpa"].apply(lambda x: 1 if x >= 3 else 0)

    # Probabilidad admision para caso rango = 1
    rank = 1
    admit_conditional = ConditionalModel(
        data[["admit"]].to_numpy(), data[["rank"]].to_numpy()
    )
    classes, probs = admit_conditional.calculate_conditional()
    index = np.where(classes)[rank - 1][0]
    print("Probability of rejection given a student studied in a rank 1 school:")
    print(1 - probs[index])

    # Probabilidad admision para caso rango = 2, gre = 450 y gpa = 3.5
    condition = [0, 1, 2]
    admit_conditional = ConditionalModel(
        data[["admit"]].to_numpy(), data[["gre", "gpa", "rank"]].to_numpy()
    )
    classes, probs = admit_conditional.calculate_conditional()
    index = np.where(list(map(lambda x: np.all(x), classes == condition)))[0][0]
    print(
        "Probability of admission given a student studied in a rank 2 school, got a gre of 450 and a gpa of 3.5:"
    )
    print(probs[index])


if __name__ == "__main__":
    main()
