import pandas as pd
import pydotplus
from sklearn.tree import DecisionTreeClassifier         # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split    # Import train_test_split function
from sklearn import metrics                             # Import scikit-learn metrics module for accuracy calculation
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

def train(X_train, y_train):
    """
    Creating instance of DecisionTreeClassifier and
    building a decision tree classifier from the training set (X, y)
    :param X_train: The training input samples.
    :param y_train: The target values (class labels) as integers or strings.
    :return: Fitted estimator.
    """
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    return clf

def split_dataset(X, y):
    """
    Splitting arrays and matrices into random train and test subsets.
    :param X: The 2D array of features.
    :param y: The array of targets variables.
    :return: List containing train-test split of inputs.
    """
    return train_test_split(X, y)

def prediction(X_test, clf_object):
    """
    Predict class or regression value for X.
    For a classification model, the predicted class for each sample in X is returned.
    :param X_test: The testing subset.
    :param clf_object: Instance of fitted DecisionTreeClassifier
    :return: The predicted classes, or the predict values.
    """
    return clf_object.predict(X_test)

def cal_accuracy(y_test, y_pred):
    """
    Accuracy classification score.
    In multilabel classification, this function computes subset accuracy:
    the set of labels predicted for a sample must exactly match the corresponding set of labels in y_test.
    :param y_test: Ground truth (correct) labels.
    :param y_pred: Predicted labels, as returned by a classifier.
    """
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

def show_plot(clf_object, feature_cols, file):
    """
    Exporting a decision tree,loading graph as defined by data in DOT format
    and saving result into file.
    :param clf_object: Instance of training DecisionTreeClassifier.
    :param feature_cols: List of features names.
    :param file: The name of the file where the plot will be saved.
    """
    dot_data = export_graphviz(clf_object,
                               feature_names = feature_cols,
                               special_characters=True,
                               rounded=True,
                               filled=True)

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

    clf = train(X_train, y_train)

    print("Result:")
    y_pred = prediction(X_test, clf)
    cal_accuracy(y_test, y_pred)
    show_plot(clf, feature_cols, 'result.png')

main()