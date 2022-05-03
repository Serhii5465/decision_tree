import pandas as pd
import pydotplus
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.tree import export_graphviz

def decode_label(data):
    """
    Encoding target labels with value between 0 and n_classes-1.
    :param data: Dataset from .csv file
    """
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

def train(X_train, y_train, criterion):
    """
    Creating instance of DecisionTreeClassifier and
    building a decision tree classifier from the training set (X, y)
    :param X_train: The training input samples.
    :param y_train: The target values (class labels) as integers or strings.
    :param criterion: The function to measure the quality of a split.
    Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.
    :return: Fitted estimator.
    """
    clf = DecisionTreeClassifier(criterion)
    clf.fit(X_train, y_train)
    return clf

def split_dataset(X, y):
    """
    Splitting arrays and matrices into random train and test subsets.
    :param X: The 2D array of features variables.
    :param y: The target variable.
    :return: List containing train-test split of inputs.
    """
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
    #print(data)
    decode_label(data)
    #print(data)

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

    print(data[feature_cols])

    # X, y = data[feature_cols], data['HeartDisease']  # Features, target variable
    # X_train, X_test, y_train, y_test = split_dataset(X, y)
    #
    # clf_gini = train(X_train, y_train, "gini")
    # clf_entropy = train(X_train, y_train, "entropy")
    #
    # print("Results Using Gini Index:")
    # y_pred_gini = prediction(X_test, clf_gini)
    # cal_accuracy(y_test, y_pred_gini)
    # show_plot(clf_gini, feature_cols, 'result_gini.png')
    #
    # print("Results Using Entropy:")
    # y_pred_entropy = prediction(X_test, clf_entropy)
    # cal_accuracy(y_test, y_pred_entropy)
    # show_plot(clf_entropy, feature_cols, 'result_entropy.png')

main()