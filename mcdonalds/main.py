import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

class ML:

    def __init__(self, dataframe):
        self.df = dataframe

    def load_data(self):
        self.df = pd.read_csv('mcdonalds/India_Menu.csv')

    def analyze(self):
        print(self.df.head(),"\n")
        print(self.df.shape,"\n")
        print(self.df.columns,"\n")
        print(self.df.describe(),"\n")
        print(self.df.info(),"\n")
        print(self.df.isnull().sum(),"\n")

    def fill_missing(self):
        self.df['Sodium (mg)'] = self.df['Sodium (mg)'].fillna(self.df['Sodium (mg)'].mean())

    def drop_unnecessary(self):
        # Drop unnecessary columns
        self.df.drop(['Menu Category', 'Menu Items'],axis=1,inplace=True)

    def transform(self):
        # Feature transformation for Per Serve Size column and convert it to float
        self.df['Per Serve Size'] = self.df['Per Serve Size'].str.replace(' ml','')
        self.df['Per Serve Size'] = self.df['Per Serve Size'].str.replace(' g','')
        self.df['Per Serve Size'] = self.df['Per Serve Size'].astype(float)

    def model_training(self):
        # Split the data into train and test
        y = self.df['Energy (kCal)']
        X = self.df.drop(['Energy (kCal)'],axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

        # XGBoost
        xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
                        max_depth = 5, alpha = 10, n_estimators = 10)
        xg_reg.fit(X_train,y_train)
        preds = xg_reg.predict(X_test)
        print("Accuracy: ",xg_reg.score(X_test, y_test), "\n")

        # Visualize the feature importance
        xgb.plot_importance(xg_reg)
        plt.rcParams['figure.figsize'] = [5, 5]
        plt.show()

        # Hyperparameter tuning
        # Parameter grid
        params = {
                'learning_rate': [0.01, 0.1, 0.2, 0.3],
                'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
                'min_child_weight': [1, 2, 3, 4, 5],
                'gamma': [0.0, 0.1, 0.2 , 0.3, 0.4],
                'colsample_bytree': [0.3, 0.4, 0.5, 0.7]
                }

        # Randomized search
        random_search = RandomizedSearchCV(xg_reg, param_distributions=params, n_iter=5, scoring='neg_mean_squared_error', n_jobs=-1, cv=5, verbose=3)
        random_search.fit(X_train, y_train)

        # Best parameters
        print(random_search.best_params_, "\n")

        # XGBoost with best parameters
        xg_reg = random_search.best_estimator_
        xg_reg.fit(X_train,y_train)
        preds = xg_reg.predict(X_test)
        print("Accuracy: ",xg_reg.score(X_test, y_test))

        # Visualize the feature importance
        xgb.plot_importance(xg_reg)
        plt.rcParams['figure.figsize'] = [5, 5]
        plt.show()


if __name__ == "__main__":
    ml = ML(None)
    ml.load_data()
    ml.analyze()
    ml.fill_missing()
    ml.drop_unnecessary()
    ml.transform()
    ml.model_training()
