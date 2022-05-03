import pandas as pd
import pydotplus
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.tree import export_graphviz

file = 'data.csv'
data = pd.read_csv(file)
# print(data.head())

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

#split dataset in features and target variable
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

X = data[feature_cols] # Features
y = data['HeartDisease'] # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y)
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()  # Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)  #Predict the response for test dataset
y_pred = clf.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

dot_data = export_graphviz(clf, feature_names = feature_cols, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png('result.png')

