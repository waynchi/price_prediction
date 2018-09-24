#!/usr/bin/python3

import json
import requests
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from pprint import pprint
from datetime import datetime
import random

def get_price_history(url, appid_arr, region, names):
    # payload = {'appid': '49520', 'cc': 'us'}
    if len(appid_arr) < 0:
        print ("empty appid array")
        return None

    payload = {'appid': 0, 'cc': region}
    price_history_all = []
    added_appids = []

    i = 0
    while i < min(10, len(appid_arr)):
        appid = random.choice(appid_arr)
        while appid in added_appids:
            appid = random.choice(appid_arr)
        added_appids.append(appid)
        payload['appid'] = str(appid)

        r = requests.get(url, params=payload)
        price_history = r.json()['data']['final']
        if len(price_history) is 0:
            continue
        formatted = r.json()['data']['formatted']
        # Add time from release date
        # Add Month of Year (for reference to holidays and events)
        for arr in price_history:
            t = arr[0]/1000
            dt = datetime.utcfromtimestamp(t)
            arr.append(dt.timetuple().tm_yday)

        # Add discount as last value
        for arr in price_history:
            t = arr[0]
            arr.append(formatted[str(t)]['discount'])

        price_history_all.extend(price_history)
        i += 1

    pprint(added_appids)

    dataset = pandas.DataFrame(price_history_all)
    dataset.drop(dataset.columns[1], axis=1, inplace=True) # drop price in favor of discount
    dataset.drop(dataset.columns[0], axis=1, inplace=True) # drop time in favor of doy
    dataset.columns = names

    return dataset

def graph(dataset):
    """
    Graphing is very useful to determining the structure of data
    """
    # box and whisker plots
    dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
    plt.show()

    # histographs
    dataset.hist()
    plt.show()

    # scatter plot matrix
    scatter_matrix(dataset)
    plt.show()

def verify_mode(X_train, Y_train, X_validation, Y_validation):
    # Make predictions on validation dataset
    knn = KNeighborsClassifier()
    knn.fit(X_train, Y_train)
    predictions = knn.predict(X_validation)
    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))


def train_model(X_train, Y_train, scoring, seed):
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))

    # evaluate each model in turn
    results = []
    names = []
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print (msg)

def main():
    # Get app id list
    r = requests.get('http://api.steampowered.com/ISteamApps/GetAppList/v2')
    applist = r.json()['applist']['apps']
    appid_arr = [a['appid'] for a in applist]

    # Load dataset
    url = 'https://steamdb.info/api/GetPriceHistory'
    region='us'
    names=['doy', 'discount']
    dataset = get_price_history(url, appid_arr, region, names)

    print(dataset.shape)

    # head
    print(dataset.head(20))

    # descriptions
    print(dataset.describe())

    # class distribution
    # print(dataset.groupby('price').size())

    # graph
    # graph(dataset)

    # Split-out validation dataset
    array = dataset.values
    pprint(array)
    X = array[:,0:len(names)-1]
    pprint(X)
    Y = array[:,len(names)-1]
    pprint(Y)
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

    # # Test options and evaluation metric
    seed = 7
    scoring = 'accuracy'

    train_model(X_train, Y_train, scoring, seed)
    verify_mode(X_train, Y_train, X_validation, Y_validation)

if __name__ == "__main__":
    main()
