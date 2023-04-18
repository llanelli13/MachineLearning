import pandas as pd
import numpy as np
#import seaborn as sns


def prepartation_donnee():

    #ouvrir fichier csv
    data_x = pd.read_csv('Data/Data_X.csv')
    data_y = pd.read_csv("Data/Data_Y.csv")

    # Fusionner les données d'entrée et de sortie en fonction de l'ID
    merged_data = pd.merge(data_x, data_y, on='ID')

    print(merged_data)

    # Supprime ligne entièrement vide
    data_notnull=merged_data.dropna(how='all')
    print(data_notnull)

    # met O sur case null
    data_dim_fill = data_notnull.fillna(0)
    print(data_dim_fill)


"""
def jspcestquoilafonctionmaistqt():
    # Charger les données d'entrée Data X et de sortie Data Y
    data_x = pd.read_csv("Data/Data_X.csv")


    # Fusionner les données d'entrée et de sortie en fonction de l'ID


    # Vérifier s'il y a des valeurs manquantes
    print(merged_data.isna().sum())

    # Analyse exploratoire des données (EDA)
    sns.pairplot(merged_data)

    # Normalisation des données
    normalized_data = (merged_data - merged_data.mean()) / merged_data.std()
    print(normalized_data.head())
"""

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    prepartation_donnee()