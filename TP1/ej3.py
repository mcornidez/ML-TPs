from conditionalModel import ConditionalModel 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():

    df = pd.read_csv("binary.csv")
    data = df[["admit", "gre", "gpa", "rank"]]
    data["gre"] = data["gre"].apply(lambda x: 1 if x >= 500 else 0)
    data["gpa"] = data["gpa"].apply(lambda x: 1 if x >= 3 else 0)

    admit_conditional = ConditionalModel(data[["admit"]].to_numpy(), data[["rank"]].to_numpy())
    #gre_conditional = Conditional(data[["gre"]].to_numpy(), data[["rank"]].to_numpy())
    #gpa_conditional = Conditional(data[["gpa"]].to_numpy(), data[["rank"]].to_numpy())



if __name__ == "__main__":
    main()
