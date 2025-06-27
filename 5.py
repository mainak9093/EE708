import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt
X = pd.DataFrame({
    "x1": [1, 2, 3, 4, 5, 6, 7, 8],
    "x2": [5, 6, 8, 10, 12, 15, 18, 20]
})
y = [10, 12, 15, 18, 21, 25, 28, 30]
tree = DecisionTreeRegressor(random_state=0)
tree.fit(X, y)

plt.figure(figsize=(10, 6)) 
plot_tree(tree, feature_names=["x1", "x2"], rounded=True)
plt.title("Fully-grown regression tree for Q2 data")
plt.tight_layout()
plt.show()
