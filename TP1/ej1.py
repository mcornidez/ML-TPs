from naiveBayes import Classifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    df = pd.read_excel("PreferenciasBritanicos.xlsx")

    prefs = df[["scones", "cerveza", "wiskey", "avena", "futbol"]].to_numpy()
    nac = df["Nacionalidad"].to_numpy()

    classifier = Classifier(prefs, nac)

    print(classifier.classify([1,0,1,1,0]))



if __name__ == "__main__":
    main()
