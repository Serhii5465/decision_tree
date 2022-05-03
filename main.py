import pandas as pd
import pydotplus
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.tree import export_graphviz

def decode_label(data):
    gender = {'M': 0, 'F': 1}
    data['Sex'] = data['Sex'].map(gender)

    chest_pain_type = {'TA': 0, 'ATA': 1, 'NAP': 2, 'ASY': 3}
    data['ChestPainType'] = data['ChestPainType'].map(chest_pain_type)

    resting_ecg = {'Normal': 0, 'ST': 1, 'LVH': 2}
    data['RestingECG'] = data['RestingECG'].map(resting_ecg)

    exercise_angina = {'N': 0, 'Y': 1}
    data['ExerciseAngina'] = data['ExerciseAngina'].map(exercise_angina)

    st_slope = {'Up': 0, 'Flat': 1, 'Down': 2}
    data['ST_Slope'] = data['ST_Slope'].map(st_slope)

def train_using_gini(X_train, y_train):
    clf_gini = DecisionTreeClassifier(criterion = "gini")
    clf_gini.fit(X_train,y_train)
    return clf_gini

def train_using_entropy(X_train, y_train):
    clf_entropy  = DecisionTreeClassifier(criterion = "entropy")
    clf_entropy .fit(X_train, y_train)
    return clf_entropy

def split_dataset(X, y):
    return train_test_split(X, y)

def prediction(X_test, clf_object):
    return clf_object.predict(X_test)

def cal_accuracy(y_test, y_pred):
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

def show_plot(clf_object, feature_cols, file):
    dot_data = export_graphviz(clf_object, feature_names = feature_cols)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png(file)

def main():
    file = 'data.csv'
    data = pd.read_csv(file)
    decode_label(data)

    feature_cols = ['Age',
                    'Sex',
                    'ChestPainType',
                    'RestingBP',
                    'Cholesterol',
                    'FastingBS',
                    'RestingECG',
                    'MaxHR',
                    'ExerciseAngina',
                    'Oldpeak',
                    'ST_Slope']

    X, y = data[feature_cols], data['HeartDisease']  # Features, target variable
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    clf_gini = train_using_gini(X_train, y_train)
    clf_entropy = train_using_entropy(X_train, y_train)

    print("Results Using Gini Index:")
    y_pred_gini = prediction(X_test, clf_gini)
    cal_accuracy(y_test, y_pred_gini)
    show_plot(clf_gini, feature_cols, 'result_gini.png')

    print("Results Using Entropy:")
    y_pred_entropy = prediction(X_test, clf_entropy)
    cal_accuracy(y_test, y_pred_entropy)
    show_plot(clf_entropy, feature_cols, 'result_entropy.png')


main()