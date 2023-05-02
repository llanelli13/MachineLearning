import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
import scipy
from prettytable import PrettyTable
from matplotlib import pyplot


def prepartation_donnee(merged_data):

    # Supprime ligne entièrement vide
    data_notnull = merged_data.dropna(how='all')
    # met O sur case null
    data_dim_fill = data_notnull.fillna(0)

    # Normalisation des données à partir de la 4ème colonne
    cols_to_normalize = data_dim_fill.columns[3:]
    data_dim_fill[cols_to_normalize] = (data_dim_fill[cols_to_normalize] - data_dim_fill[cols_to_normalize].mean()) / data_dim_fill[cols_to_normalize].std()

    # remplace FR et DE par des float
    data_dim_fill['COUNTRY'] = data_dim_fill['COUNTRY'].map({'FR': 1, 'DE': 2})

    return data_dim_fill

def analyse_exploratoire(df):

    print("\nInformation sur nos variables : ")
    df.info()
    print(df.describe())

    print("\nPossibilité de voir les histogrammes, diagramme en boite pour chaque variable")
    print("Mais aussi voir graphique de dispersion entre chaque variable et le prix")
    # Histogramme
    """
    for col in df.columns:
        plt.hist(df[col])
        plt.title(col)
        plt.show()
    """
    # Diagramme en boite
    """
    for col in df.columns:
        plt.boxplot(df[col])
        plt.title(col)
        plt.show()
    """
    #Graphique de dispersion
    """
    for col in df.columns:
        plt.scatter(df[col], df['TARGET'], marker='+')
        plt.xlabel(col)
        plt.ylabel('PRIX')
        plt.title('Graphique de dispersion entre ' + str(col) + ' et le Prix')
        plt.show()
    """

    #Matrice corrélation

    print("------------------------------------------------------------------------------------")
    print("\nVisualisation de la matrice de corrélation : ")
    corr_matrix = df.corr()
    #Afficher la matrice de corrélation sous forme de heatmap
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
    plt.show()

    print("\nVisualisation de la partie inferieur de la matrice de corrélation sans la diagonal: ")
    # Enlever la partie inferieur de la matrice
    mask = np.tril(np.ones_like(corr_matrix, dtype=bool))
    corr_matrix = corr_matrix.mask(mask)
    # Afficher la matrice de corrélation sans diagonale
    sns.heatmap(corr_matrix, cmap="coolwarm", annot=False, fmt=".2f")
    plt.show()

    print("\nÉtude de nos variables corrélé positivement ")
    # Corrélation positive
    tab_f = []
    indices = np.where(corr_matrix > 0.75)
    print("\nForte corrélation positive entre : ")
    colonne = indices[0]
    ligne = indices[1]
    column_names = df.columns
    for k in range(len(colonne)):
        column_names_c = column_names[colonne[k]]
        column_names_l = column_names[ligne[k]]
        tab_f.append([column_names_c, column_names_l])
        print(column_names_c, column_names_l)

    # Graphique de dispersion corrélation positive
    for k in range(len(tab_f)):
        plt.scatter(df[tab_f[k][0]], df[tab_f[k][1]], marker='+')
        plt.xlabel(tab_f[k][0])
        plt.ylabel(tab_f[k][1])
        plt.title('Graphique de dispersion entre valeurs corrélées positivement')
        plt.show()

    print("\nÉtude de nos variables corrélé négativement")
    # Corrélation négative
    tab_n=[]
    indices = np.where(corr_matrix < -0.75)
    print("\nForte corrélation négative entre : ")
    colonne = indices[0]
    ligne = indices[1]
    column_names = df.columns
    for k in range(len(colonne)):
        column_names_c = column_names[colonne[k]]
        column_names_l = column_names[ligne[k]]
        tab_n.append([column_names_c, column_names_l])
        print(column_names_c, column_names_l)

    # Graphique de dispersion corrélation négative
    for k in range(len(tab_n)):
        plt.scatter(df[tab_n[k][0]], df[tab_n[k][1]], marker='+')
        plt.xlabel(tab_n[k][0])
        plt.ylabel(tab_n[k][1])
        plt.title('Graphique de dispersion entre valeurs corrélées négativement')
        plt.show()

def regression_lineaire(df):
    x_test = df.iloc[:, 3:35]
    y_test = df["TARGET"]

    # Instance and fit
    lr_model = LinearRegression()
    lr_model.fit(x_test, y_test)

    print('\nRegression linéaire : ')
    # Score
    score_lr = lr_model.score(x_test, y_test)

    y_pred = lr_model.predict(x_test)

    #Visualiser performance du model
    performance = pd.DataFrame(
        {'True Value': np.exp(y_test), 'Prediction': np.exp(y_pred), 'Error': y_test - y_pred})
    print(performance)

    #Calcul du score et erreur quandratique
    mse = np.mean((y_test - y_pred) ** 2)
    rmse = np.sqrt(mse)

    print("Score " + str(score_lr))
    print('Erreur quadratique :', rmse)

    print("Peut aller plus loin dans l'analyse des résultats en visualisant la corrélation de Sperman")
    """
    res, pvalue = scipy.stats.spearmanr(x_test, y_test)
    print("Corrélation de Spearman: ", res)
    print("P-valeur: ", pvalue)
    """

    return score_lr

def regression_lineaire_regularise(df):
    x_test = df.iloc[:, 3:35]
    y_test = df["TARGET"]

    print('\nRegression linéaire régularisée RIDGE: ')
    # Ridge
    rid = Ridge(alpha=0.1)
    rid.fit(x_test, y_test)

    y_pred=rid.predict(x_test)

    # Visualiser performance du model
    performance = pd.DataFrame({'True Value': np.exp(y_test), 'Prediction': np.exp(y_pred),'Error':y_test-y_pred})
    print(performance)

    # Calcul du score et erreur quandratique
    mse = np.mean((y_test - y_pred) ** 2)
    rmse = np.sqrt(mse)

    score_RD=r2_score(y_test, rid.predict(x_test))
    print("Score :", score_RD)
    print('Erreur quadratique :', rmse)

    print("Peut aller plus loin dans l'analyse des résultats en visualisant la corrélation de Sperman")
    """
    res, pvalue = scipy.stats.spearmanr(x_test, y_test)
    print("Corrélation de Spearman: ", res)
    print("P-valeur: ", pvalue)
    """

    #Lasso
    # Créer un objet Lasso en spécifiant le coefficient de régularisation alpha
    reg = Lasso(alpha=0.1)

    print('\nRegression linéaire régularisée LASSO: ')
    # Adapter le modèle à l'aide des données d'entraînement
    reg.fit(x_test, y_test)

    # Utiliser le modèle pour faire des prédictions sur les données de test
    y_pred = reg.predict(x_test)

    # Visualiser performance du model
    performance = pd.DataFrame(
        {'True Value': np.exp(y_test), 'Prediction': np.exp(y_pred), 'Error': y_test - y_pred})
    print(performance)

    # Calcul du score et erreur quandratique
    mse = np.mean((y_test - y_pred) ** 2)
    rmse = np.sqrt(mse)
    score_LA=r2_score(y_test, reg.predict(x_test))
    print("Score :", score_LA)
    print('Erreur quadratique :', rmse)

    print("Peut aller plus loin dans l'analyse des résultats en visualisant la corrélation de Sperman")
    """
    res, pvalue = scipy.stats.spearmanr(x_test, y_test)
    print("Corrélation de Spearman: ", res)
    print("P-valeur: ", pvalue)
    """
    return score_RD, score_LA

def knn (df):

    x = df.iloc[:, 3:35]
    y = df["TARGET"]

    # Création d'un classificateur k-NN avec k=3
    knn = KNeighborsRegressor(n_neighbors=5).fit(x,y)

    print('\nMéthode des k plus proche voisin: ')
    #score
    score_knn=knn.score(x,y)

    # Données de test
    #X_test = df.iloc[:, [3, 4]].values

    # Prédiction des classes pour les données de test
    y_pred = knn.predict(x)

    # Visualiser performance du model
    performance=pd.DataFrame({'True Value': y, 'Prediction':y_pred,'Error':y-y_pred})
    print(performance)

    # Calcule l'erreur quadratique moyenne (RMSE) entre les valeurs réelles et les valeurs prédites.
    mse = np.mean((y - y_pred) ** 2)
    rmse = np.sqrt(mse)

    print("Score :", score_knn)
    print('Erreur quadratique :', rmse)

    print("Peut aller plus loin dans l'analyse des résultats en visualisant la corrélation de Sperman")
    """
    res, pvalue = scipy.stats.spearmanr(x, y)
    print("Corrélation de Spearman: ", res)
    print("P-valeur: ", pvalue)
    """
    return score_knn

def arbre_decision(df):
    x = df.iloc[:, 3:35]
    y = df["TARGET"]
    print('\nArbre de décision pour la régression: ')


    model = DecisionTreeRegressor(max_depth=100)
    model.fit(x, y)

    y_pred = model.predict(x)
    performance = pd.DataFrame({'True Value': y, 'Prediction': y_pred, 'Error': y - y_pred})
    print(performance)

    mse = np.mean((y - y_pred) ** 2)
    rmse = np.sqrt(mse)

    score_ab = model.score(x, y)
    print("Score :", score_ab)
    print('Erreur quadratique :', rmse)

    print("Peut aller plus loin dans l'analyse des résultats en visualisant la corrélation de Sperman")
    """
    res, pvalue=scipy.stats.spearmanr(x, y)
    print("Corrélation de Spearman: ", res)
    print("P-valeur: ", pvalue)
    """

    return score_ab


def display_feat_imp_reg(df):
    x = df.iloc[:, 3:35]
    y = df["TARGET"]
    model = DecisionTreeRegressor(max_depth=100)
    model.fit(x, y)

    # get importance
    importance = model.feature_importances_

    # summarize feature importance
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))
    # plot feature importance
    pyplot.bar([x for x in range(len(importance))], importance)
    pyplot.show()

    #print(df.iloc[:, 18])
    print("\nFR_NUCLEAR influence majoritairement le prix\n")

    """
    print(df.iloc[:, 6])
    print(df.iloc[:, 14])
    print(df.iloc[:, 21])
    print(df.iloc[:, 23])
    print(df.iloc[:, 29])
    """
    print("On a également FR_DE_EXCHANGE, FR_COAL, DE_WINDPOW, DE_LIGNITE et FR_WIND qui influence le prix de manière importante  \n")

    """
    print(df.iloc[:, 5])
    print(df.iloc[:, 10])
    """

    print("Contrairement à DE_FR_EXCHANGE et FR_NET_IMPORT qui influence très peu le prix")


if __name__ == '__main__':

    #ouvrir fichier
    data_x = pd.read_csv('Data/Data_X.csv')
    data_y = pd.read_csv("Data/Data_Y.csv")

    # Fusionner les données d'entrée et de sortie en fonction de l'ID
    merged_data = pd.merge(data_x, data_y, on='ID')

    print("\nNos données\n")
    print(merged_data)
    #Preparation des données

    df=prepartation_donnee(merged_data)
    print("\nNos données normalisées\n")
    print(df)

    print("------------------------------------------------------------------------------------")

    #Analyser les données
    analyse_exploratoire(df)

    print("------------------------------------------------------------------------------------")

    #Modélisation des données
    print("\nModélisation des données : \n")
    score_lr=regression_lineaire(df)
    score_RE, score_LA=regression_lineaire_regularise(df)
    score_knn=knn(df)
    score_ab=arbre_decision(df)
    score_comp = [['Reg linéaire','Reg linéraire RIDGE','Reg linéraire LASSO' ,'Knn','Arbre de décision'], [score_lr,score_RE,score_LA,score_knn,score_ab]]

    # Création de la table
    table = PrettyTable()

    # Ajout des colonnes
    table.add_column('Méthode',['Reg linéaire', 'Reg linéraire RIDGE', 'Reg linéraire LASSO', 'Knn', 'Arbre de décision'])
    table.add_column('Score', [score_lr, score_RE, score_LA, score_knn, score_ab])

    # Affichage de la table
    print(table)

    print("Le score maximum est obtenue pour la méthode Arbre de décision")
    print("\nOn va donc utiliser l'arbre de déciion pour definir DataNew_Y à partir de DataNew_X")

    print("------------------------------------------------------------------------------------")

    display_feat_imp_reg(df)

    """
    #Fait les prévisions avec le modèle le plus performant
    x_new= pd.read_csv('Data/DataNew_X.csv')

    # Preparation des données
    df_new = prepartation_donnee(x_new)

    x = df.iloc[:, 3:35]
    y = df["TARGET"]

    model = DecisionTreeRegressor(max_depth=100)
    model.fit(x, y)

    x_new_nor=df_new.iloc[:, 3:35]

    y_pred = model.predict(x_new_nor)

    print("\nPrix estimé avec l'arbre de décision pour la régression:", y_pred)
    """




