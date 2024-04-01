import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


class ML:

    def __init__(self, dataframe):
        self.df = dataframe
    
    def partition(self, x_label, y_label):
        df_arr = self.df.loc[:,[x_label, y_label]].values
        return df_arr

    def kmeans(self, n_cluster, arr):
        kmeans = KMeans(n_clusters=n_cluster)
        kmeans.fit(arr)
        labels = kmeans.labels_
        self.df['labels'] = labels
        print(self.df['labels'].value_counts(),"\n")

    def gmm(self, n_component, arr):
        gmm = GaussianMixture(n_components=n_component)
        gmm.fit(arr)
        labels = gmm.predict(arr)
        self.df['labels'] = labels
        print(self.df['labels'].value_counts(),"\n")

    def visualize(self, x_label, y_label):
        sns.scatterplot(x=x_label, y=y_label, hue='labels', data=self.df)
        plt.show()



# Load the data
df = pd.read_csv('nba/game_scores.csv')

# Analysis of the data
#print(df.head(),"\n")
#print(df.shape,"\n")
#print(df.columns,"\n")
#print(df.describe(),"\n")
#print(df.info(),"\n")
#print(df.isnull().sum(),"\n")


# PREPROCESSING

# Drop unnecessary columns
df.drop(['HOME_FGA','HOME_FG3A', 'HOME_FTA'],axis=1,inplace=True)
df.drop(['AWAY_FGA','AWAY_FG3A', 'AWAY_FTA'],axis=1,inplace=True)
df.drop(['DATE'],axis=1,inplace=True)

# Encode the team names. -1 for away 1 for home 0 for others
# get team names
teams = df['HOME'].unique()
teams = np.append(teams, df['AWAY'].unique())

# add team names as new columns to the dataframe
for team in teams:
    df[team] = 0

# encode the team names
for i in range(len(df)):
    df.loc[i, df.loc[i, 'HOME']] = 1
    df.loc[i, df.loc[i, 'AWAY']] = -1

# drop the team name columns
df.drop(['HOME', 'AWAY'], axis=1, inplace=True)


# Encode the SEASON column
df['SEASON'] = df['SEASON'].apply(lambda x: x[:4])
df = pd.get_dummies(df)



if __name__ == "__main__":
    # Create the ML object
    ml = ML(df)
    
    # Cluster based on the scores
    arr = ml.partition('HOME_PTS', 'AWAY_PTS')
    ml.kmeans(3, arr)
    ml.visualize('HOME_PTS', 'AWAY_PTS')
    ml.gmm(3, arr)
    ml.visualize('HOME_PTS', 'AWAY_PTS')
    
    # Cluster based on the steal
    arr = ml.partition('HOME_STL', 'AWAY_STL')
    ml.kmeans(3, arr)
    ml.visualize('HOME_STL', 'AWAY_STL')
    ml.gmm(3, arr)
    ml.visualize('HOME_STL', 'AWAY_STL')
    
    # Cluster based on the turnover
    arr = ml.partition('HOME_TURNOVERS', 'AWAY_TURNOVERS')
    ml.kmeans(3, arr)
    ml.visualize('HOME_TURNOVERS', 'AWAY_TURNOVERS')
    ml.gmm(3, arr)
    ml.visualize('HOME_TURNOVERS', 'AWAY_TURNOVERS')