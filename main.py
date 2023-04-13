import pandas as pd
import numpy as np

def prepartation_donnee():
    #ouvrir fichier csv
    data = pd.read_csv('Data/Data_X.csv.csv')
    print(data)
    # Supprime ligne entièrement vide
    data.dropna(how='all')
    print(data)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    prepartation_donnee()