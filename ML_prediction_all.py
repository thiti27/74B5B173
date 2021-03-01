import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import export_graphviz
import os
from sklearn import tree

# Load the Train data
df = pd.read_csv("train_Rev.csv")
df_test = pd.read_csv("test_Rev06.csv").fillna(0)


#Explore the data
print(df.keys())
# print(breast_ca["feature_names"])
# print(breast_ca["target_names"])
# print(breast_ca.target)


# Split the data into train and test set
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
X_train = np.array(df.drop(['result_name'], axis=1))
y_train = np.array(df['result_name'])
X_test = np.array(df_test.drop(['result_name'], axis=1))
y_test = np.array(df_test['result_name'])

print("X_train ", X_train.shape)
print("X_test ", X_test.shape)
print("y_train ", y_train.shape)
print("y_test ", y_test.shape)

# Feature scaling for
X_scaler = StandardScaler().fit(X_train)
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)


# Train and fit the model: SVC with Gaussian RBF Kernal   --------------------------------
svc_clf = SVC(kernel="rbf", gamma="auto", C=1, max_iter=-1).fit(X_train_scaled, y_train)
print("SVC with Gaussian RBF Kernel model:")
print("Train set accuracy = " + str(svc_clf.score(X_train_scaled, y_train)))
print("Test set accuracy = " + str(svc_clf.score(X_test_scaled, y_test)))
print("-----------------------------------------------------------------------")

# Train and fit the model: Decision tree
tree_clf = DecisionTreeClassifier(max_depth=None).fit(X_train, y_train)
print("\nDecision Tree model:")
print("Train set accuracy = " + str(tree_clf.score(X_train, y_train)))
print("Test set accuracy = " + str(tree_clf.score(X_test, y_test)))
print("\nImportance of each feature:\n", tree_clf.feature_importances_) # Determine features importances
print("-----------------------------------------------------------------------")

# Train and fit the model: random forest classifier
rforest_clf = RandomForestClassifier().fit(X_train, y_train)
print("\nRandom Forest model:")
print("Train set accuracy = " + str(rforest_clf.score(X_train, y_train)))
print("Test set accuracy = " + str(rforest_clf.score(X_test, y_test)))
print("\nImportance of each feature:\n", rforest_clf.feature_importances_)

fn = df.columns
cn = np.unique(df['result_name'])
fig, axes = plt.subplots(nrows = 1,ncols = 5,figsize = (10,2), dpi=900)
for index in range(0, 5):
    tree.plot_tree(rforest_clf.estimators_[index],
                   feature_names = fn, 
                   class_names=cn,
                   filled = True,
                   ax = axes[index]);

    axes[index].set_title('Estimator: ' + str(index), fontsize = 11)
fig.savefig('rf_5trees.png')
print("-----------------------------------------------------------------------")

# Train and fit the model: AdaBoost classifier
aDaboost_clf = AdaBoostClassifier().fit(X_train, y_train)
print("\nAda Boost model:")
print("Train set accuracy = " + str(aDaboost_clf.score(X_train, y_train)))
print("Test set accuracy = " + str(aDaboost_clf.score(X_test, y_test)))
print("\nImportance of each feature:\n", aDaboost_clf.feature_importances_)
print("-----------------------------------------------------------------------")

# Train and fit the model: Gradient Boosting classifier
gBoost_clf = GradientBoostingClassifier(n_estimators=200).fit(X_train, y_train)
print("\nGradient Boosting classifier model:")
print("Train set accuracy = " + str(gBoost_clf.score(X_train, y_train)))
print("Test set accuracy = " + str(gBoost_clf.score(X_test, y_test)))
print("\nImportance of each feature:\n", gBoost_clf.feature_importances_)
print("-----------------------------------------------------------------------")




# Export graph -------------------------------------------------------------
# export_graphviz(tree_clf, out_file="breast_cancer.dot", feature_names=breast_ca["feature_names"],
#     class_names=breast_ca["target_names"], rounded=True, filled=True)

# os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'
# os.system('dot -Tpng breast_cancer.dot -o breast_cancer.png')



# X_selected = breast_ca.data[:, [7,20]]  # เลือกตำแหน่งที่ 7 และ 20
# print(X_selected) 
# print(X_selected[:2,:]) 