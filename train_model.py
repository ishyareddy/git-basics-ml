from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Load data
X, y = load_iris(return_X_y=True)

# Create and train model
clf = LogisticRegression(max_iter=200)
clf.fit(X, y)

print("Model trained")
