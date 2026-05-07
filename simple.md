## 1. Constraint Satisfaction Problems (CSP)
```python
from ortools.sat.python import cp_model
model = cp_model.CpModel()

# Variables
A = model.new_int_var(0, 5, 'A')
B = model.new_int_var(0, 5, 'B')
C = model.new_int_var(0, 5, 'C')

# Constraints
model.add(A != 2)
model.add_all_different([A, B, C])

# Conditional (If A=1 then B=3)
flag = model.new_bool_var('flag')
model.add(A == 1).only_enforce_if(flag)
model.add(A != 1).only_enforce_if(flag.Not())
model.add(B == 3).only_enforce_if(flag)

# Solve
solver = cp_model.CpSolver()
solver.solve(model)
```

## 2. Adversarial Search (Minimax with Alpha-Beta Pruning)
```python
class Node:
    def __init__(self, val=None, kids=[]):
        self.val, self.kids = val, kids
    def is_leaf(self): return not self.kids

def ab_prune(node, d, a, b, is_max):
    if d == 0 or node.is_leaf(): return node.val
    
    if is_max:
        v = float('-inf')
        for k in node.kids:
            v = max(v, ab_prune(k, d-1, a, b, False))
            a = max(a, v)
            if b <= a: break
        return v
    else:
        v = float('inf')
        for k in node.kids:
            v = min(v, ab_prune(k, d-1, a, b, True))
            b = min(b, v)
            if b <= a: break
        return v
```

## 3. Exploratory Data Analysis (EDA) & Visualization
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Pandas Basics
df = pd.read_csv('data.csv')
df.head(); df.info(); df.describe(); df.isnull().sum()
df['X'] = df['X'].fillna(df['X'].median())
df = pd.get_dummies(df.drop('Y', axis=1), columns=['Z'])
pd.crosstab(df['A'], df['B'])

# Matplotlib
plt.plot(x, y)          # Line
plt.scatter(x, y)       # Scatter
plt.bar(cats, vals)     # Bar
plt.hist(x, bins=10)    # Histogram
plt.show()

# Seaborn (Relational & Categorical)
sns.lineplot(x=x, y=y, data=df)
sns.scatterplot(x=x, y=y, data=df)
sns.barplot(x=cat, y=num, data=df)
sns.countplot(x=cat, data=df)
sns.boxplot(x=cat, y=num, data=df)
sns.violinplot(x=cat, y=num, data=df)

# Seaborn (Distributions)
sns.histplot(data, kde=True)
sns.displot(data)
sns.kdeplot(data)
sns.ecdfplot(data)
```

## 4. Supervised Learning
```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

# Split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)

# Decision Tree
dt = DecisionTreeClassifier(max_depth=4).fit(X_tr, y_tr)
dt_acc = accuracy_score(y_te, dt.predict(X_te))

# Support Vector Machine (SVM)
svm = SVC(kernel='rbf').fit(X_tr, y_tr)
svm_acc = accuracy_score(y_te, svm.predict(X_te))

# Linear Regression
lr = LinearRegression().fit(X_tr, y_tr)
lr_preds = lr.predict(X_te)
```

## 5. Unsupervised Learning (K-Means)
```python
from sklearn.cluster import KMeans

# Elbow Method to find optimal K
wcss = []
for k in range(1, 11):
    km = KMeans(n_clusters=k).fit(X)
    wcss.append(km.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.show()

# Train K-Means & Predict
km = KMeans(n_clusters=3).fit(X)
labels = km.predict(X)
cents = km.cluster_centers_

# Visualize Clusters & Centroids
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(cents[:, 0], cents[:, 1], c='red', marker='X', s=200)
plt.show()
```
