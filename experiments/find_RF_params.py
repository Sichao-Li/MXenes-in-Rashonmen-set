from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing
from src.utilities import load_data
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
import sys
elemental = sys.argv[1]
'''
Search the best parameters for the random forest
'''
if __name__ == "__main__":
    X_df = load_data(filename='../data/data_withLattice.csv')
    X_features = X_df.iloc[:, 1:-5]
    feature_names = X_features.columns.values
    y_multilabel = X_df.iloc[:, -5:]
    label_names = y_multilabel.columns.values

    data_index = []
    for index, row in X_features.iterrows():
        indices = [i for i, x in enumerate(row) if x == 1]
        indices[1] = indices[1] - 9
        indices[2] = indices[2] - 11
        indices[3] = indices[3] - 16
        data_index.append(indices)
    X_features_classes = np.array(data_index)
    X = X_features.to_numpy()
    y = y_multilabel.to_numpy()
    scaler = preprocessing.MinMaxScaler()
    y = scaler.fit_transform(y)
    if elemental:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X_features_classes, y, test_size=0.2, random_state=42)

    model_RFR = RandomForestRegressor(random_state=42, max_features=None,n_jobs=-1, criterion='squared_error',min_impurity_decrease=1e-07)
    parameters = {'max_depth':(10, 11, 12, 13, 14), 'min_samples_leaf':(1, 2, 3, 4, 5), 'min_samples_split':(1, 2, 3, 4, 5), 'n_estimators':(120, 150, 170, 200)}
    model_best = GridSearchCV(model_RFR, parameters, cv=10)
    model_best.fit(X_train, y_train)
    print(model_best.best_params_)