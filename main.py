import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def prepartation_donnee():

    #ouvrir fichier csv
    data_x = pd.read_csv('Data/Data_X.csv')
    data_y = pd.read_csv("Data/Data_Y.csv")

    # Fusionner les données d'entrée et de sortie en fonction de l'ID
    merged_data = pd.merge(data_x, data_y, on='ID')

    # Supprime ligne entièrement vide
    data_notnull=merged_data.dropna(how='all')

    # met O sur case null
    data_dim_fill = data_notnull.fillna(0)

    # remplace FR et DE par des float
    data_dim_fill['COUNTRY'] = data_dim_fill['COUNTRY'].map({'FR': 1, 'DE': 2})

    # Calculer la matrice de corrélation
    corr_matrix = data_dim_fill.corr()

    # Afficher la matrice de corrélation sous forme de heatmap
    #sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
    #plt.show()

    # Enlever la partie inferieur de la matrice
    mask = np.tril(np.ones_like(corr_matrix, dtype=bool))
    corr_matrix = corr_matrix.mask(mask)

    # Afficher la matrice de corrélation sans diagonale
    #sns.heatmap(corr_matrix, cmap="coolwarm", annot=False, fmt=".2f")
    #plt.show()

    indices = np.where(corr_matrix > 0.75)

    print("forte coorélation entre : ")


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