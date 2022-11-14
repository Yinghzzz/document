import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures 
from sklearn.metrics import r2_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Lasso, Ridge
from sklearn.impute import KNNImputer
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import accuracy_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
import pandas as pd
import sklearn

#load the dataset
X_train = pd.read_csv('X_train.csv').drop(['id'], axis=1)
y_train = pd.read_csv('y_train.csv')["y"]
X_test = pd.read_csv('X_test.csv').drop(['id'], axis=1)
#ds1d = X_train.describe()
#Use IQR to detect the outliers  

#drop the useless featrues
sel=VarianceThreshold()
X_train = sel.fit_transform(X_train)
X_test = sel.transform(X_test)



#imputation of the missing value(KNN imputer)
imp = KNNImputer(n_neighbors=5)
X_train_imputed = imp.fit_transform(X_train)
X_test_imputed = imp.fit_transform(X_test)

# standardization
scaler = StandardScaler()
X_train_st = scaler.fit_transform(X_train_imputed)
X_test_st = scaler.transform(X_test_imputed)

## for feature selection 
best_alpha = 0.388
clf = Lasso(alpha=best_alpha)
clf.fit(X_train_st,y_train)
weights = clf.coef_
important_features = []
for i in range(len(weights)):
    if weights[i] != 0:
        important_features.append(i)
print("Number of important features for our model:", len(important_features))
print("Indexes of important features for our model:", important_features)

#load the dataset again
X_train = pd.read_csv('X_train.csv').drop(['id'], axis=1)
y_train = pd.read_csv('y_train.csv')["y"]
X_test = pd.read_csv('X_test.csv').drop(['id'], axis=1)

#drop the useless featrues
sel=VarianceThreshold()
X_train = sel.fit_transform(X_train)
X_test = sel.transform(X_test)
X_train = X_train[:,important_features]
X_test = X_test[:,important_features]
    

#imputation of the missing value(iterative)
imp = KNNImputer(n_neighbors=5)
X_train_imputed = imp.fit_transform(X_train)
X_test_imputed = imp.fit_transform(X_test)

#isolisation forest
clf = IsolationForest(contamination=0.03)
pred = clf.fit_predict(X_train_imputed)
index_outliers = []
for i in range(len(pred)):
    if pred[i] == -1:
        index_outliers.append(i)
X_train_imputed = np.delete(X_train_imputed, index_outliers, axis=0)

y_train = np.delete(y_train.values, index_outliers, axis=0)


#outliers
ds_train = pd.DataFrame(X_train).describe()
ds_test = pd.DataFrame(X_test).describe()
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
for index, row in ds_train.iteritems():
    index_outliers = []
    IQR  = row["75%"] - row["25%"]
    lower_limit = row["25%"] - 1.5*IQR
    upper_limit = row["75%"] + 1.5*IQR
    X_train[index][(X_train[index]<=lower_limit) | (X_train[index]>=upper_limit)] = X_train[index].median()
for index, row in ds_test.iteritems():
    index_outliers = []
    IQR  = row["75%"] - row["25%"]
    lower_limit = row["25%"] - 1.5*IQR
    upper_limit = row["75%"] + 1.5*IQR
    X_test[index][(X_test[index]<=lower_limit) | (X_test[index]>=upper_limit)] = X_test[index].median()

# standardization
scaler = StandardScaler()
X_train_st = scaler.fit_transform(X_train_imputed)
X_test_st = scaler.transform(X_test_imputed)

#score test(Gradient Boosting Regressor)
regressor = RandomForestRegressor()
scores = cross_val_score(regressor, X_train_st, y_train, cv=5, scoring= 'r2')
print(scores.mean(),scores.std())

regressor.fit(X_train_st, y_train)
y_pred = regressor.predict(X_test_st)

y_pred = pd.DataFrame(y_pred)
y_pred.to_csv("y_predict.csv")
















