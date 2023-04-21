import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score


def prepartation_donnee():
    # ouvrir fichier csv
    data_x = pd.read_csv('Data/Data_X.csv')
    data_y = pd.read_csv("Data/Data_Y.csv")

    # Fusionner les données d'entrée et de sortie en fonction de l'ID
    merged_data = pd.merge(data_x, data_y, on='ID')

    # Supprime ligne entièrement vide
    data_notnull = merged_data.dropna(how='all')
    # met O sur case null
    data_dim_fill = data_notnull.fillna(0)

    # Normalisation des données à partir de la 4ème colonne
    cols_to_normalize = data_dim_fill.columns[3:]
    data_dim_fill[cols_to_normalize] = (data_dim_fill[cols_to_normalize] - data_dim_fill[cols_to_normalize].mean()) / data_dim_fill[cols_to_normalize].std()

    # remplace FR et DE par des float
    data_dim_fill['COUNTRY'] = data_dim_fill['COUNTRY'].map({'FR': 1, 'DE': 2})

    # Calculer la matrice de corrélation
    corr_matrix = data_dim_fill.corr()
    # Afficher la matrice de corrélation sous forme de heatmap
    # sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
    # plt.show()
    # Enlever la partie inferieur de la matrice
    mask = np.tril(np.ones_like(corr_matrix, dtype=bool))
    corr_matrix = corr_matrix.mask(mask)
    # Afficher la matrice de corrélation sans diagonale
    # sns.heatmap(corr_matrix, cmap="coolwarm", annot=False, fmt=".2f")
    # plt.show()

    # Corrélation positive
    tab_f = []
    indices = np.where(corr_matrix > 0.75)
    print("\nForte corrélation positive entre : ")
    colonne = indices[0]
    ligne = indices[1]
    column_names = data_dim_fill.columns
    for k in range(len(colonne)):
        column_names_c = column_names[colonne[k]]
        column_names_l = column_names[ligne[k]]
        tab_f.append([column_names_c,column_names_l])
        print(column_names_c, column_names_l)

    # corrélation négative
    tab_n=[]
    indices = np.where(corr_matrix < -0.75)
    print("\nForte corrélation négative entre : ")
    colonne = indices[0]
    ligne = indices[1]
    column_names = data_dim_fill.columns
    for k in range(len(colonne)):
        column_names_c = column_names[colonne[k]]
        column_names_l = column_names[ligne[k]]
        tab_n.append([column_names_c, column_names_l])
        print(column_names_c, column_names_l)

    # Sous-ensemble
    #print('\n',tab_f)
    #print(tab_n)
    return data_dim_fill,tab_f, tab_n

"""
    # Avoir le maximum d'une colonne
    max = 0
    for i in range(len(data_dim_fill)):
        data = data_dim_fill['FR_NET_IMPORT'][i]
        if data > max:
            max = data
    print("max = ", round(max,2))

    # Avoir le minimum d'une colonne
    min = 0
    for i in range(len(data_dim_fill)):
        data = data_dim_fill['FR_NET_IMPORT'][i]
        if data < min:
            min = data
    print("min = ", round(min, 2))    
    
    # Avoir la moyenne d'une colonne
    data_sum = 0
    for value in data_dim_fill['FR_NET_IMPORT']:
        data_sum += value
    data_mean = data_sum / len(data_dim_fill)
    print("moyenne = ", round(data_mean,2))
"""

def analyse_exploratoire(df,tab_f,tab_n):
    #df.info()
    #print(df.describe())
    print(df)

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
    print(tab_f)
    """
    for k in range (len(tab_f)):
        plt.scatter(df[tab_f[k][0]],df[tab_f[k][1]],marker='+')
        plt.xlabel(tab_f[k][0])
        plt.ylabel(tab_f[k][1])
        plt.title('Graphique de dispersion entre valeurs corrélées positivement')
        plt.show()

    for k in range (len(tab_n)):
        plt.scatter(df[tab_n[k][0]],df[tab_n[k][1]],marker='+')
        plt.xlabel(tab_n[k][0])
        plt.ylabel(tab_n[k][1])
        plt.title('Graphique de dispersion entre valeurs corrélées négativement')
        plt.show()
    
    for col in df.columns:
        plt.scatter(df[col], df['TARGET'], marker='+')
        plt.xlabel(col)
        plt.ylabel('PRIX')
        plt.title('Graphique de dispersion entre ' + str(col) + ' et le Prix')
        plt.show()
    """

    #Matrice corrélation
    corr_matrix = df.corr()
    #Afficher la matrice de corrélation sous forme de heatmap
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
    #plt.show()
    # Enlever la partie inferieur de la matrice
    mask = np.tril(np.ones_like(corr_matrix, dtype=bool))
    corr_matrix = corr_matrix.mask(mask)
    # Afficher la matrice de corrélation sans diagonale
    sns.heatmap(corr_matrix, cmap="coolwarm", annot=False, fmt=".2f")
    #plt.show()

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
def regression_lineaire(df):
    for col in df.columns:
        X = df[col]
        Y = df["TARGET"]

        axes = plt.axes()
        plt.xlabel(col)
        plt.ylabel('PRIX')
        axes.grid()
        plt.scatter(X, Y)
        # plt.show()

        # res = stats.linregress(X, Y)
        # Coefficient de determination
        # print(f"R-squared: {res.rvalue**2:.6f}")

        slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)

        # Plot the data along with the fitted line:
        a = df[col][2]

        plt.plot(X, Y, 'o', label='original data')
        plt.plot(X, intercept + slope * X, 'r', label='fitted line')
        plt.legend()
        plt.show()

        print(slope * a + intercept)  # check si cest proche de la vraie valeur

        # Parcourir les colonnes
        # Prendre une ligne temoin et
        #
        # #
def regression_lineaire_regularise(df):
    x_test = df.iloc[:, 3:35]
    y_test = df["TARGET"]
    print(x_test)
    print(y_test)

    rid = Ridge(10).fit(x_test, y_test)
    print(r2_score(y_test, rid.predict(x_test)))
    """
    

    #x_log = np.log(x_test)
    #y_log = np.log(y_test)

    x_test = x_test.replace([np.inf, -np.inf], np.nan)
    y_test = y_test.replace([np.inf, -np.inf], np.nan)

    x_test = x_test.dropna()
    y_test = y_test.dropna()

    model = HistGradientBoostingRegressor().fit(x_test, y_test)
    # Instance and fit
    #lrLog_model = LinearRegression().fit(x_log, y_log)
    # Remove zeroes
    X_test_log = x_test[(x_test.x > 0) & (x_test.y > 0) & (x_test.z > 0) ]
    y_test_log = y_test[X_test_log.index]
    # Log Transform X_test and y test
    X_test_log = np.log(X_test_log)
    y_test_log = np.log(y_test_log)
    # Score
    score_log = model.score(X_test_log, y_test_log)
    print(score_log)
    # Predictions
    preds = model.predict(X_test_log)
    # Performance
    performance=pd.DataFrame({ 'True Value': np.exp(y_test_log), 'Prediction': np.exp(preds)}).head(5)
    print(performance)
    """
def knn (df):

        x = df.iloc[:, 3:35]
        print(x)
        y = df["TARGET"]
        print(y)

        # Création d'un classificateur k-NN avec k=3
        knn = KNeighborsRegressor(n_neighbors=5).fit(x,y)

        #score
        score_knn=knn.score(x,y)
        print(score_knn)

        # Données de test
        #X_test = df.iloc[:, [3, 4]].values

        # Prédiction des classes pour les données de test
        y_pred = knn.predict(x)

        performance=pd.DataFrame({'True Value': y, 'Prediction':y_pred,'Error':y-y_pred})

        print(performance)
        # Affichage des prédictions



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    sousensemble=prepartation_donnee()
    analyse_exploratoire(sousensemble[0],sousensemble[1],sousensemble[2])
    #regression_lineaire(sousensemble[0])
    regression_lineaire_regularise(sousensemble[0])
    #knn(sousensemble[0])