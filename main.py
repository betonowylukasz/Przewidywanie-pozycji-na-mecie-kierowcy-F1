import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

from functions import add_last_position, download_in_range, dict_to_dataframe, add_standings, driver_standings_download, team_name_fixup


def x_test_dataset():
    drivers = ['Verstappen', 'Perez', 'Alonso', 'Stroll', 'Hamilton', 'Russell', 'Leclerc', 'Bearman', 'Ocon',
               'Gasly', 'Norris', 'Piastri', 'Hulkenberg', 'Magnussen', 'Bottas', 'Zhou', 'Tsunoda', 'Ricciardo',
               'Albon', 'Sargeant']
    dict = {"age": [27, 34, 43, 26, 39, 26, 27, 19, 28, 28, 25, 23, 37, 32, 35, 25, 24, 35, 28, 24],
            "constructor": ['red_bull', 'red_bull', 'aston_martin', 'aston_martin', 'mercedes', 'mercedes', 'ferrari',
                            'ferrari', 'alpine', 'alpine', 'mclaren', 'mclaren', 'haas', 'haas', 'sauber', 'sauber',
                            'rb', 'rb', 'williams', 'williams'],
            "standing": [1, 2, 9, 10, 7, 5, 4, 11, 17, 18, 6, 8, 16, 12, 19, 11, 14, 13, 15, 20],
            "last_result": [1, 2, 9, 10, 7, 5, 4, 11, 17, 18, 6, 8, 16, 12, 19, 11, 14, 13, 15, 20],
            "grid": [1, 3, 4, 10, 8, 7, 2, 11, 17, 18, 6, 5, 15, 13, 16, 20, 9, 14, 12, 19]}
    return pd.DataFrame.from_dict(dict), drivers


def create_categorical_feature(dataset, column_to_modify):
    categorical = pd.get_dummies(dataset[column_to_modify], prefix=column_to_modify)
    dataset.drop(column_to_modify, axis=1, inplace=True)
    dataset = pd.concat([dataset, categorical], axis=1)
    return dataset

def split_datasets(dataset):
    results_cols = ['position']
    prediction_cols = [col for col in dataset.columns if col not in results_cols]
    return dataset[prediction_cols], dataset[results_cols]

def find_best_parameters(model, parameters, X, y, cv=10, verbose=1, n_jobs=-1):
    grid_object = GridSearchCV(model, parameters, scoring=make_scorer(accuracy_score), cv=cv, verbose=verbose, n_jobs=n_jobs)
    grid_object = grid_object.fit(X, y)
    return grid_object.best_estimator_

def main():
    dataset=download_in_range(2017, 2024)
    dataset=add_last_position(dataset)
    dataset=add_standings(dataset)
    dataset=dict_to_dataframe(dataset)

    for i in range(len(dataset)):
        if dataset['grid'].values[i]==0: dataset['grid'].values[i]=20

    dataset = team_name_fixup(dataset)

    columns_to_drop = ['season', 'round', 'circuit', 'driver']
    dataset.drop(columns_to_drop, axis=1, inplace=True)

    for i in range(len(dataset)):
        dataset['position'].values[i] -= 1

    x_train, y_train = split_datasets(dataset)
    x_test, drivers = x_test_dataset()

    length=len(x_test.index)

    merge_dataset = pd.concat([x_train, x_test])
    merge_dataset = create_categorical_feature(merge_dataset, 'constructor')
    x_train = merge_dataset.iloc[:-length]
    x_test = merge_dataset.iloc[-length:]

    y_train = y_train.values.ravel()

    sc = StandardScaler()

    sc.fit(x_train)

    x_train_sc = sc.transform(x_train)
    x_test_sc = sc.transform(x_test)

    # random_forest = RandomForestClassifier()
    # parameters = {'n_estimators': [350],
    #               'max_depth': [7],
    #               'min_samples_split': [3],
    #               'random_state': [i for i in range(101)]
    #               }
    # random_forest = find_best_parameters(random_forest, parameters, x_train_sc, y_train)
    # print(random_forest)

    # clf = SVC(kernel='rbf', gamma=0.015, C=24)
    # clf.fit(x_train_sc, y_train)
    # svm_pred = clf.predict(x_test_sc)

    rf = RandomForestClassifier(n_estimators=350, random_state=13, max_depth=7, min_samples_split=3)
    rf.fit(x_train_sc, y_train)
    rf_pred = rf.predict(x_test_sc)

    # svm_pred = [pred + 1 for pred in svm_pred]
    rf_pred=[pred+1 for pred in rf_pred]

    # for driver, position in zip(drivers, svm_pred):
    #     print(f"{driver}: {position}")
    #
    # print(f"\n")

    for driver, position in zip(drivers, rf_pred):
        print(f"{driver}: {position}")

if __name__=='__main__': main()
