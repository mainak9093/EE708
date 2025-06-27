import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

data = {
    "price":       ["low", "low", "low", "low", "med", "med", "med", "med", "high", "high", "high"],
    "maintenance": ["low", "med", "low", "high", "med", "med", "high", "high", "med",  "high", "high"],
    "capacity":    [2, 4, 4, 4, 4, 4, 2, 5, 4, 2, 5],
    "airbag":      ["no", "yes", "no", "no", "no", "yes", "yes", "no", "yes", "yes", "yes"],
    "profitable":  ["yes", "no", "yes", "yes", "no", "yes", "no", "yes", "yes", "no", "yes"],
}

df = pd.DataFrame(data)

X = pd.get_dummies(df[["price", "maintenance", "capacity", "airbag"]], drop_first=False)
y = df["profitable"].map({"no": 0, "yes": 1})

clf = DecisionTreeClassifier(random_state=0)
clf.fit(X, y)

plt.figure(figsize=(12, 8))  
plot_tree(
    clf,
    feature_names=X.columns,
    class_names=["no", "yes"],
    filled=True,
    rounded=True,
)
plt.title("Fully-grown classification tree for Q4 data")
plt.tight_layout()
plt.show()

print("Confusion matrix (training data):")
print(confusion_matrix(y, clf.predict(X)))
print("\nDetailed classification report:")
print(classification_report(y, clf.predict(X), target_names=["no", "yes"]))
