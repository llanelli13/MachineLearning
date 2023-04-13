import pandas as pd
import numpy as np

def prepartation_donnee():
    #ouvrir fichier csv
    data = pd.read_csv('Data/Data_X.csv')
    print(data)
    # Supprime ligne entièrement vide
    data_notnull=data.dropna(how='all')
    print(data)

    # met O sur case null
    data_dim_fill = data_notnull.fillna(0)
    print(data_dim_fill)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    prepartation_donnee()