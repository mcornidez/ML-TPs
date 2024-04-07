from conditionalModel import ConditionalModel 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    df = pd.read_excel("PreferenciasBritanicos.xlsx")

    prefs = df[["scones", "cerveza", "wiskey", "avena", "futbol"]].to_numpy()
    nac = df["Nacionalidad"].transform(lambda x: 1 if x == 'E' else 0).to_numpy()

    conditional = ConditionalModel(prefs, nac)

    print(conditional.naive_classify(np.array([0,0,1,1,0])))




if __name__ == "__main__":
    main()
