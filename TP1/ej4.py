from conditionalModel import ConditionalModel
import pandas as pd
import numpy as np

def main(): 
    df = pd.read_csv("Data/binary.csv")
    data = df[["admit", "gre", "gpa", "rank"]]
    data["gre"] = data["gre"].apply(lambda x: 1 if x >= 500 else 0)
    data["gpa"] = data["gpa"].apply(lambda x: 1 if x >= 3 else 0)
    total_count = len(df)

    gres = list(data["gre"])
    gre_counts = np.zeros(2)
    for i in range(2):
        gre_counts[i] = gres.count(i)
    gre_probs = gre_counts/total_count

    gpas = list(data["gpa"])
    gpa_counts = np.zeros(2)
    for i in range(2):
        gpa_counts[i] = gpas.count(i)
    gpa_probs = gpa_counts/total_count

    comps = data[["gre", "gpa"]].to_numpy()
    comp_counts = np.zeros(4)
    comp_counts[0] = np.array(list(map(lambda x: 1 if (x == [0,0]).all() else 0, comps))).sum()
    comp_counts[1] = np.array(list(map(lambda x: 1 if (x == [0,1]).all() else 0, comps))).sum()
    comp_counts[2] = np.array(list(map(lambda x: 1 if (x == [1,0]).all() else 0, comps))).sum()
    comp_counts[3] = np.array(list(map(lambda x: 1 if (x == [1,1]).all() else 0, comps))).sum()

    comp_probs = comp_counts/total_count
    print("P(gre = 0 ^ gpa = 0) =", comp_probs[0], "& P(gre = 0) * P(gpa = 0) = ", round(gre_probs[0] * gpa_probs[0], 7))
    print("P(gre = 0 ^ gpa = 1) =", comp_probs[1], "& P(gre = 0) * P(gpa = 1) = ", round(gre_probs[0] * gpa_probs[1], 7))
    print("P(gre = 1 ^ gpa = 0) =", comp_probs[2], " & P(gre = 1) * P(gpa = 0) = ", round(gre_probs[1] * gpa_probs[0], 7))
    print("P(gre = 1 ^ gpa = 1) =", comp_probs[3], "   & P(gre = 1) * P(gpa = 1) = ", round(gre_probs[1] * gpa_probs[1], 7))

if __name__ == "__main__":
    main()