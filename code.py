import re
import pandas as pd
import pyttsx3
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import csv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load the training and testing data
training = pd.read_csv('Training.csv')
testing = pd.read_csv('Testing.csv')

# Preparing the data
cols = training.columns
cols = cols[:-1]
x = training[cols]
y = training['prognosis']
y1 = y

# Reduced data for symptoms per disease
reduced_data = training.groupby(training['prognosis']).max()

# Encoding string labels to numeric values
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
testx = testing[cols]
testy = testing['prognosis']
testy = le.transform(testy)

# DecisionTreeClassifier Model
clf1 = DecisionTreeClassifier()
clf = clf1.fit(x_train, y_train)

# Cross-validation to check accuracy
scores = cross_val_score(clf, x_test, y_test, cv=3)
print(f"Decision Tree Classifier score: {scores.mean()}")

# Support Vector Classifier Model
model = SVC()
model.fit(x_train, y_train)
print("SVM score: ", model.score(x_test, y_test))

# Feature importance from the Decision Tree
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
features = cols

# Text-to-speech initialization
def readn(nstr):
    engine = pyttsx3.init()
    engine.setProperty('voice', "english+f5")
    engine.setProperty('rate', 130)
    engine.say(nstr)
    engine.runAndWait()
    engine.stop()

# Dictionaries to store severity, description, and precaution data
severityDictionary = dict()
description_list = dict()
precautionDictionary = dict()
symptoms_dict = {}

# Populating the symptoms dictionary with symptom indices
for index, symptom in enumerate(x):
    symptoms_dict[symptom] = index

# Calculate the severity of conditions based on symptoms
def calc_condition(exp, days):
    sum = 0
    for item in exp:
        sum += severityDictionary[item]
    if (sum * days) / (len(exp) + 1) > 13:
        print("You should consult a doctor.")
    else:
        print("It may not be severe, but you should take precautions.")

# Load symptom description data from CSV
def getDescription():
    global description_list
    with open('symptom_Description.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            _description = {row[0]: row[1]}
            description_list.update(_description)

# Load symptom severity data from CSV
def getSeverityDict():
    global severityDictionary
    with open('Symptom_severity.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        try:
            for row in csv_reader:
                _diction = {row[0]: int(row[1])}
                severityDictionary.update(_diction)
        except:
            pass

# Load symptom precautions data from CSV
def getprecautionDict():
    global precautionDictionary
    with open('symptom_precaution.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            _prec = {row[0]: [row[1], row[2], row[3], row[4]]}
            precautionDictionary.update(_prec)


# Function to collect user info and greet
def getInfo():
    print("-----------------------------------AI Medical ChatBot-----------------------------------")
    print("\nYour Name? \t\t\t\t", end="-> ")
    name = input("")
    print("Hello, ", name)


# Function to check for a pattern match in symptoms
def check_pattern(dis_list, inp):
    pred_list = []
    inp = inp.replace(' ', '_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list = [item for item in dis_list if regexp.search(item)]
    if len(pred_list) > 0:
        return 1, pred_list
    else:
        return 0, []


# Function for secondary prediction
def sec_predict(symptoms_exp):
    df = pd.read_csv('Training.csv')
    X = df.iloc[:, :-1]
    y = df['prognosis']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)

    rf_clf = DecisionTreeClassifier()
    rf_clf.fit(X_train, y_train)

    symptoms_dict = {symptom: index for index, symptom in enumerate(X)}
    input_vector = np.zeros(len(symptoms_dict))

    for item in symptoms_exp:
        input_vector[[symptoms_dict[item]]] = 1

    return rf_clf.predict([input_vector])


# Function to print the disease prediction
def print_disease(node):
    node = node[0]
    val = node.nonzero()
    disease = le.inverse_transform(val[0])
    return list(map(lambda x: x.strip(), list(disease)))


# Recursive function to traverse the decision tree and predict the disease
def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    chk_dis = ",".join(feature_names).split(",")
    symptoms_present = []

    while True:
        print("\nEnter the symptom you are experiencing  \t\t", end="-> ")
        disease_input = input("")
        conf, cnf_dis = check_pattern(chk_dis, disease_input)
        if conf == 1:
            print("Searches related to input: ")
            for num, it in enumerate(cnf_dis):
                print(num, ") ", it)
            if num != 0:
                print(f"Select the one you meant (0 - {num}):  ", end="")
                conf_inp = int(input(""))
            else:
                conf_inp = 0

            disease_input = cnf_dis[conf_inp]
            break
        else:
            print("Enter a valid symptom.")

    while True:
        try:
            num_days = int(input("Okay. For how many days? : "))
            break
        except:
            print("Enter a valid input.")

    def recurse(node, depth):
        indent = "  " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]

            if name == disease_input:
                val = 1
            else:
                val = 0
            if val <= threshold:
                recurse(tree_.children_left[node], depth + 1)
            else:
                symptoms_present.append(name)
                recurse(tree_.children_right[node], depth + 1)
        else:
            present_disease = print_disease(tree_.value[node])
            red_cols = reduced_data.columns
            symptoms_given = red_cols[reduced_data.loc[present_disease].values[0].nonzero()]
            print("Are you experiencing any of these symptoms?")
            symptoms_exp = []
            for syms in list(symptoms_given):
                inp = ""
                print(syms, "? : ", end='')
                while True:
                    inp = input("")
                    if inp == "yes" or inp == "no":
                        break
                    else:
                        print("Please provide proper answers (yes/no) : ", end="")
                if inp == "yes":
                    symptoms_exp.append(syms)

            second_prediction = sec_predict(symptoms_exp)
            calc_condition(symptoms_exp, num_days)
            if present_disease[0] == second_prediction[0]:
                print("You may have ", present_disease[0])
                print(description_list[present_disease[0]])
            else:
                print("You may have ", present_disease[0], " or ", second_prediction[0])
                print(description_list[present_disease[0]])
                print(description_list[second_prediction[0]])

            precution_list = precautionDictionary[present_disease[0]]
            print("Take the following precautions: ")
            for i, j in enumerate(precution_list):
                print(i + 1, ") ", j)

    recurse(0, 1)


# Load the data for severity, description, and precautions
getSeverityDict()
getDescription()
getprecautionDict()

# Get user information
getInfo()

# Start the disease prediction process
tree_to_code(clf, cols)

print("----------------------------------------------------------------------------------------")
