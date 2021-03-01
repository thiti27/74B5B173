import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.tree import export_graphviz
import os
from sklearn import tree

# Load the Train data
df = pd.read_csv("Data Train_HumTemp T8 Rev_00.csv")

# Load the Test data
df_test = pd.read_csv("Data Test_HumTemp T8 Rev_00.csv").fillna(0)

X_train = np.array(df.drop(["result_name"], axis=1))
y_train = np.array(df["result_name"])
X_test = np.array(df_test.drop(["result_name"], axis=1))
y_test = np.array(df_test["result_name"])

# Feature scaling for
X_scaler = StandardScaler().fit(X_train)
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)

# Train and fit the model: random forest classifier
rforest_clf = RandomForestClassifier().fit(X_train, y_train)
print("\nRandom Forest model:")
print("Train set accuracy = " + str(rforest_clf.score(X_train, y_train)))
print("Test set accuracy = " + str(rforest_clf.score(X_test, y_test)))
print("\nImportance of each feature:\n", rforest_clf.feature_importances_)

# Prediction X_test
y_predict = rforest_clf.predict(X_test)

# Prediction by value
# print(rforest_clf.predict([[49.30647659,22.77714157,12.00732422,0,1,1]]))

# Export predict and compare data to csv
data = ""
for col in df_test.columns:
    data = data + "," + col

pipelineObject = {
    data: df_test.values.tolist(),
    "Predict": rforest_clf.predict(X_test),
    "Compare": rforest_clf.predict(X_test) == y_test,
}
df = pd.DataFrame(pipelineObject, columns=[data, "Predict", "Compare"])
df.to_csv(r"export_dataframe.csv", index=False, header=True)

# Plot non-normalized confusion matrix
from sklearn.metrics import plot_confusion_matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(rforest_clf , X_test, y_test,

                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)
    # print(title)
    # print(disp.confusion_matrix)
plt.show()