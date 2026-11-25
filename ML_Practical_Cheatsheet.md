# Machine Learning Practical Exam – Copy-Paste Cheat Sheet (M.Sc.IT Sem III)

Each practical = one section. Each code block = one notebook cell, in journal order. Paste top to bottom.
Every cell that needs a data file **creates a small sample automatically if the file is missing**, so nothing breaks in the exam.

**Cell 0 – run once at the start (Google Colab / Jupyter)**
```python
%pip install -q statsmodels xgboost hmmlearn bayesian-optimization pyro-ppl pymc arviz fastapi uvicorn requests openpyxl xlrd
```
(Most of these are already in Colab. Needed by: statsmodels → P3.2, bayesian-optimization → P4c, xgboost → P4f, hmmlearn → P5.2, pyro-ppl / pymc / arviz → P6.1, fastapi / uvicorn → P10.)

**What I fixed compared with the journal (so you can explain it if asked)**
- `fillna(method='ffill')` is removed in new pandas → `.ffill()`.
- P1 outlier code mixed `Phone 1` and `Phone 2`; now each column uses its own quartiles.
- P3 class was written `def init` → must be `def __init__`; GIF saved with `writer='pillow'` (ffmpeg not needed).
- P3.2 VIF was computed including the target column; now only on the features.
- P4a plot labelled column 8 as "Age" (age is column 0).
- P4f XGBoost: `preds.astype(int)` turned every probability into 0 (79.64 % = just the majority class); now a 0.5 threshold is used.
- P4c: missing imports (`randint`, `RandomizedSearchCV`, `cross_val_score`) added; tree is now visualised.
- P6.1: `pymc3` is dead → `pymc`; `sd=` → `sigma=`; `shade=True` → `fill=True`.
- P8: posterior mean forgot to divide by σ² and "sigma" was really the precision; fixed (μ posterior is now ≈ 5, as it should be).
- P9: `def init` → `def __init__`.

---

## Practical 1 – Data pre-processing and exploration

### 1.1 Load CSV, handle missing values, inconsistent formatting, outliers

```python
import os
import numpy as np
import pandas as pd

path = '/content/customers-100.csv' if os.path.isdir('/content') else 'customers-100.csv'   # change if your file is elsewhere
if not os.path.exists(path):                  # sample data so the cell always runs
    rng = np.random.default_rng(0)
    n = 100
    pd.DataFrame({
        'Index': range(1, n + 1),
        'Customer Id': [f'C{i:05d}' for i in range(n)],
        'First Name': rng.choice(['Sheryl', 'Preston', 'Roy', 'Linda', None], n),
        'Country': rng.choice([' Chile', 'DJIBOUTI ', 'india', ' India', 'Slovakia'], n),
        'Phone 1': [f'{rng.integers(100, 999)}-{rng.integers(100, 999)}-{rng.integers(1000, 9999)}'
                    if rng.random() > 0.1 else None for _ in range(n)],
        'Phone 2': [str(rng.integers(10**5, 10**6)) if rng.random() > 0.3 else None for _ in range(n)],
        'Subscription Date': pd.date_range('2020-01-01', periods=n, freq='7D').astype(str),
    }).to_csv(path, index=False)
    print('Sample file created at', path)

df = pd.read_csv(path)
print("Shape:", df.shape)
print("Missing values before:\n", df.isna().sum())

# 1) missing values: forward fill (journal method), then anything still empty -> 'Unknown'
df = df.ffill()
text_cols = df.select_dtypes(exclude='number').columns
df[text_cols] = df[text_cols].fillna('Unknown')

# 2) inconsistent formatting: trim whitespace, same capitalisation, proper dates
df['Country'] = df['Country'].astype(str).str.strip().str.title()
df['Subscription Date'] = pd.to_datetime(df['Subscription Date'], errors='coerce')
print("\nCountries after cleaning:", sorted(df['Country'].unique())[:8])
```

```python
# 3) outliers: keep digits only -> numeric -> cap with the IQR rule (each column uses its OWN quartiles)
for col in ['Phone 1', 'Phone 2']:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'\D', '', regex=True), errors='coerce')
    if df[col].notna().any():
        df[col] = df[col].fillna(df[col].median())
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower, upper)
        print(f"{col}: {n_out} outliers capped to [{lower:.0f}, {upper:.0f}]")
print(df.head())
```

### 1.2 Descriptive statistics, univariate and bivariate graphs, features and target

```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

path = '/content/titanic3.xls' if os.path.isdir('/content') else 'titanic3.xls'
try:
    data = pd.read_excel(path)                # .xls needs: pip install xlrd
except Exception:
    print("titanic3.xls not found - using a generated Titanic-like dataset")
    rng = np.random.default_rng(1)
    n = 800
    data = pd.DataFrame({
        'pclass': rng.integers(1, 4, n), 'survived': rng.integers(0, 2, n),
        'name': [f'Person {i}' for i in range(n)], 'sex': rng.choice(['male', 'female'], n),
        'age': np.where(rng.random(n) > 0.2, rng.normal(30, 13, n).clip(0.5, 80), np.nan),
        'sibsp': rng.integers(0, 4, n), 'parch': rng.integers(0, 3, n),
        'fare': rng.gamma(2, 15, n), 'embarked': rng.choice(['S', 'C', 'Q'], n)})

summary_stats = data.describe(include='all')
print(summary_stats)
print("\nMissing values:\n", data.isna().sum())
```

```python
# Univariate graphs
plt.figure(figsize=(10, 6))
sns.histplot(data['age'].dropna(), kde=True)
plt.title('Distribution of Age')
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x='pclass', data=data)
plt.title('Passengers per class')
plt.show()

# Bivariate graphs
plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='fare', data=data)
plt.title('Age vs fare')
plt.show()

plt.figure(figsize=(6, 4))
sns.boxplot(x='pclass', y='fare', data=data)
plt.title('Fare by class')
plt.show()

plt.figure(figsize=(6, 5))
sns.heatmap(data.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation heatmap')
plt.show()

print("Target variable : survived")
print("Possible features: pclass, sex, age, sibsp, parch, fare, embarked")
```

### 1.3 Label encoding, scaling, binarization

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, Binarizer, OneHotEncoder

np.random.seed(0)
data = pd.DataFrame({
    'Feature1': np.random.choice(['A', 'B', 'C'], 100),
    'Feature2': np.random.randn(100),
    'Feature3': np.random.randint(1, 100, 100),
    'Target': np.random.choice([0, 1], 100)})

# Label encoding
data['Feature1_encoded'] = LabelEncoder().fit_transform(data['Feature1'])

# One-hot encoding
onehot = OneHotEncoder(sparse_output=False).fit_transform(data[['Feature1']])
data[['F1_A', 'F1_B', 'F1_C']] = onehot

# Standard scaling (mean 0, std 1)
data[['Feature2_scaled', 'Feature3_scaled']] = StandardScaler().fit_transform(data[['Feature2', 'Feature3']])

# Min-max scaling (0 to 1)
data['Feature3_minmax'] = MinMaxScaler().fit_transform(data[['Feature3']])

# Binarization (value > 50 -> 1 else 0)
data['Feature3_binarized'] = Binarizer(threshold=50).fit_transform(data[['Feature3']])
print(data.head())
```

---

## Practical 2 – Testing hypothesis: FIND-S algorithm

```python
import csv

# 1) create the training dataset as a CSV file (last column = class label)
rows = [['sunny', 'warm', 'normal', 'strong', 'warm', 'same', 'yes'],
        ['sunny', 'warm', 'high', 'strong', 'warm', 'same', 'yes'],
        ['rainy', 'cold', 'high', 'strong', 'warm', 'change', 'no'],
        ['sunny', 'warm', 'high', 'strong', 'cool', 'change', 'yes']]
with open('enjoysports.csv', 'w', newline='') as f:
    csv.writer(f).writerows(rows)

# 2) read it back
data = []
with open('enjoysports.csv') as f:
    for row in csv.reader(f):
        data.append(row)
        print(row)

num_attributes = len(data[0]) - 1
print("\nThe initial hypothesis:", ['0'] * num_attributes)

# 3) FIND-S: start with the first POSITIVE example, generalise on every other positive example
hypothesis = None
for i, row in enumerate(data):
    if row[-1].lower() == 'yes':
        if hypothesis is None:
            hypothesis = row[:-1].copy()
        else:
            for j in range(num_attributes):
                if row[j] != hypothesis[j]:
                    hypothesis[j] = '?'
        print(f"After positive training instance {i}: {hypothesis}")
    else:
        print(f"Training instance {i} is negative -> ignored")

print("\nThe maximally specific hypothesis is:", hypothesis)
```

---

## Practical 3 – Linear models

### 3.1 Simple linear regression (from scratch with gradient descent)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.metrics import mean_squared_error, r2_score

url = 'https://media.geeksforgeeks.org/wp-content/uploads/20240320114716/data_for_lr.csv'
try:
    data = pd.read_csv(url)
except Exception:                      # no internet -> generated data with the same shape/columns
    rng = np.random.default_rng(0)
    xs = np.linspace(0, 100, 700)
    data = pd.DataFrame({'x': xs, 'y': xs + rng.normal(0, 3, 700)})
data = data.dropna()

x = data['x'].values.reshape(-1, 1)
y = data['y'].values.reshape(-1, 1)
train_input, train_output = x[:500], y[:500]          # training set
test_input, test_output = x[500:], y[500:]            # validation set
print("Train:", train_input.shape, "| Test:", test_input.shape)
```

```python
class LinearRegressionScratch:
    def __init__(self):
        self.parameters = {}
        self.loss = []

    def forward_propagation(self, x):
        return self.parameters['m'] * x + self.parameters['c']          # y = m x + c

    def cost_function(self, predictions, y):
        return np.mean((y - predictions) ** 2)                          # MSE

    def backward_propagation(self, x, y, predictions):
        df = predictions - y
        return {'dm': 2 * np.mean(x * df), 'dc': 2 * np.mean(df)}       # partial derivatives

    def update_parameters(self, derivatives, learning_rate):
        self.parameters['m'] -= learning_rate * derivatives['dm']
        self.parameters['c'] -= learning_rate * derivatives['dc']

    def train(self, x, y, learning_rate, iters):
        self.parameters['m'] = -np.random.uniform(0, 1)
        self.parameters['c'] = -np.random.uniform(0, 1)
        self.loss = []

        fig, ax = plt.subplots()
        x_vals = np.linspace(float(x.min()), float(x.max()), 100)
        line, = ax.plot(x_vals, self.parameters['m'] * x_vals + self.parameters['c'],
                        color='red', label='Regression Line')
        ax.scatter(x, y, marker='o', color='green', s=8, label='Training Data')
        ax.set_ylim(min(0.0, float(y.min())), float(y.max()) + 1)

        def update(frame):
            predictions = self.forward_propagation(x)
            cost = self.cost_function(predictions, y)
            derivatives = self.backward_propagation(x, y, predictions)
            self.update_parameters(derivatives, learning_rate)
            line.set_ydata(self.parameters['m'] * x_vals + self.parameters['c'])
            self.loss.append(cost)
            print("Iteration = {}, Loss = {}".format(frame + 1, cost))
            return line,

        ani = FuncAnimation(fig, update, frames=iters, interval=200, blit=True,
                            init_func=lambda: (line,), repeat=False, cache_frame_data=False)
        ani.save('linear_regression_A.gif', writer='pillow')            # saves the animation as a GIF
        plt.xlabel('Input'); plt.ylabel('Output'); plt.title('Linear Regression')
        plt.legend()
        plt.show()
        return self.parameters, self.loss
```

```python
np.random.seed(0)
model = LinearRegressionScratch()
parameters, loss = model.train(train_input, train_output, learning_rate=0.0001, iters=20)

print("\nSlope m =", round(parameters['m'], 4), "| Intercept c =", round(parameters['c'], 4))
print("Interpretation: when input increases by 1, output changes by about", round(parameters['m'], 3))

test_pred = model.forward_propagation(test_input)
print("Test MSE      :", round(mean_squared_error(test_output, test_pred), 4))
print("Test R-squared:", round(r2_score(test_output, test_pred), 4))

plt.plot(loss, 'o-'); plt.xlabel('Iteration'); plt.ylabel('Loss (MSE)'); plt.title('Training loss')
plt.show()
```

### 3.2 Multiple linear regression – feature selection and multicollinearity

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

np.random.seed(0)
size = 300
X1 = np.random.normal(0, 1, size)
X2 = X1 + np.random.normal(0, 0.1, size)                       # X2 almost equal to X1
X3 = 0.5 * X1 + 0.5 * X2 + np.random.normal(0, 0.1, size)      # X3 also correlated
Y = 2 * X1 + 3 * X2 + 1.5 * X3 + np.random.normal(0, 1, size)
df = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3, 'Y': Y})

def vif_table(frame):
    Xc = add_constant(frame)
    return pd.DataFrame({'feature': frame.columns,
                         'VIF': [variance_inflation_factor(Xc.values, i) for i in range(1, Xc.shape[1])]})

print("VIF (above 10 = serious multicollinearity):")
print(vif_table(df[['X1', 'X2', 'X3']]).round(2))
print("\nCorrelation matrix:\n", df.corr().round(2))
```

```python
features = ['X1', 'X2', 'X3']
X_train, X_test, y_train, y_test = train_test_split(df[features], df['Y'], test_size=0.2, random_state=42)

# 1) ordinary linear regression
lr_model = LinearRegression().fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# 2) Ridge regression handles multicollinearity by shrinking coefficients
ridge_model = Ridge(alpha=100).fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)

# 3) feature selection: drop the feature with the highest VIF until all VIF < 10
selected = features.copy()
while len(selected) > 1:
    v = vif_table(df[selected])
    if v['VIF'].max() < 10:
        break
    worst = v.loc[v['VIF'].idxmax(), 'feature']
    print("Dropping", worst, "(VIF =", round(v['VIF'].max(), 1), ")")
    selected.remove(worst)
print("Selected features:", selected)
lr_sel = LinearRegression().fit(X_train[selected], y_train)
y_pred_sel = lr_sel.predict(X_test[selected])

print("\nLinear Regression (all)  - MSE: {:.2f}, R2: {:.3f}".format(mean_squared_error(y_test, y_pred_lr), r2_score(y_test, y_pred_lr)))
print("Ridge Regression (all)   - MSE: {:.2f}, R2: {:.3f}".format(mean_squared_error(y_test, y_pred_ridge), r2_score(y_test, y_pred_ridge)))
print("Linear Reg (selected)    - MSE: {:.2f}, R2: {:.3f}".format(mean_squared_error(y_test, y_pred_sel), r2_score(y_test, y_pred_sel)))
print("\nCoefficients  linear:", lr_model.coef_.round(2), "| ridge:", ridge_model.coef_.round(2), "| selected:", lr_sel.coef_.round(2))
```

### 3.3 Regularised models – Ridge, LASSO, ElasticNet

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

# Boston housing data (loaded manually, as load_boston was removed from sklearn)
try:
    raw_df = pd.read_csv("http://lib.stat.cmu.edu/datasets/boston", sep=r"\s+", skiprows=22, header=None)
    values = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    cols = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
    boston_pd = pd.DataFrame(values, columns=cols)
    boston_pd['House Price'] = raw_df.values[1::2, 2]
    print("Boston dataset loaded")
except Exception:                                    # no internet -> built-in diabetes data (same code works)
    d = load_diabetes(as_frame=True)
    boston_pd = d.frame.rename(columns={'target': 'House Price'})
    print("No internet - using the diabetes dataset instead (same steps)")

print(boston_pd.head())
feature_names = boston_pd.columns[:-1]

# scatter plots: each feature vs target
plt.style.use('ggplot')
fig = plt.figure(figsize=(16, 12))
for index, name in enumerate(feature_names):
    ax = fig.add_subplot(4, 4, index + 1)
    ax.scatter(boston_pd[name], boston_pd['House Price'], s=6)
    ax.set_ylabel('House Price', size=9); ax.set_xlabel(name, size=9)
plt.tight_layout()
plt.show()
```

```python
x_train, x_test, y_train, y_test = train_test_split(boston_pd.iloc[:, :-1], boston_pd.iloc[:, -1],
                                                    test_size=0.25, random_state=0)
print("Train:", x_train.shape, y_train.shape, "| Test:", x_test.shape, y_test.shape)

models = {'Linear': LinearRegression(), 'Ridge (L2)': Ridge(alpha=1),
          'Lasso (L1)': Lasso(alpha=1), 'ElasticNet': ElasticNet(alpha=1)}
coefs, mse = {}, {}
for name, m in models.items():
    m.fit(x_train, y_train)
    pred = m.predict(x_test)
    mse[name] = np.mean((pred - y_test) ** 2)
    coefs[name] = m.coef_
    print(f"{name:<12} test MSE: {mse[name]:.3f}")

coef_table = pd.DataFrame(coefs, index=x_train.columns).round(4)
print("\nCoefficients (Lasso / ElasticNet shrink some to exactly 0 = feature selection):\n", coef_table)

coef_table.plot(kind='bar', figsize=(14, 5), title='Coefficient estimates by model')
plt.axhline(0, color='black', lw=0.8)
plt.tight_layout()
plt.show()
```

---

## Practical 4 – Discriminative models

### 4a Logistic regression – accuracy, precision, recall, ROC

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, classification_report,
                             confusion_matrix, roc_curve, auc)

diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
y_binary = (y > np.median(y)).astype(int)                 # 1 = high progression, 0 = low

X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy : {:.2f}%".format(accuracy * 100))
print("Precision: {:.3f}".format(precision_score(y_test, y_pred)))
print("Recall   : {:.3f}".format(recall_score(y_test, y_pred)))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```

```python
# scatter of two features coloured by true class (age = column 0, bmi = column 2)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_test[:, 2], y=X_test[:, 0], hue=y_test, palette={0: 'blue', 1: 'red'})
plt.xlabel("BMI (scaled)")
plt.ylabel("Age (scaled)")
plt.title("Test data by class\nAccuracy: {:.2f}%".format(accuracy * 100))
plt.legend(title="Diabetes", loc="upper right")
plt.show()

# ROC curve
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
```

### 4b k-Nearest Neighbour – read CSV, print correct and wrong predictions

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# create the CSV file, then read it back
iris = load_iris(as_frame=True)
iris.frame.to_csv('iris.csv', index=False)
df = pd.read_csv('iris.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("Correct predictions:")
for xi, t, p in zip(X_test, y_test, y_pred):
    if t == p:
        print(f"  {xi} -> predicted {names[p]} (actual {names[t]})")
print("\nWrong predictions:")
wrong = 0
for xi, t, p in zip(X_test, y_test, y_pred):
    if t != p:
        wrong += 1
        print(f"  {xi} -> predicted {names[p]} but actual {names[t]}")
if wrong == 0:
    print("  none")
print("\nAccuracy:", knn.score(X_test, y_test))
```

```python
# effect of k on train / test accuracy
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))
for i, k in enumerate(neighbors):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    train_accuracy[i] = knn.score(X_train, y_train)
    test_accuracy[i] = knn.score(X_test, y_test)

plt.plot(neighbors, test_accuracy, label='Testing dataset Accuracy')
plt.plot(neighbors, train_accuracy, label='Training dataset Accuracy')
plt.legend()
plt.xlabel('n_neighbors')
plt.ylabel('Accuracy')
plt.show()
```

### 4c Decision tree – control depth, tune hyperparameters, visualise

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import randint
from sklearn.datasets import fetch_california_housing, load_diabetes
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error

try:
    data = fetch_california_housing()
except Exception:                       # no internet -> built-in dataset (same code)
    data = load_diabetes()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# depth controls overfitting: deep tree = perfect train score, worse test score
for depth in [2, 4, 6, 10, None]:
    t = DecisionTreeRegressor(max_depth=depth, random_state=42).fit(X_train, y_train)
    print(f"max_depth={str(depth):<5} train R2={t.score(X_train, y_train):.3f}  test R2={t.score(X_test, y_test):.3f}")

# visualise a shallow tree
tree = DecisionTreeRegressor(max_depth=3, random_state=42).fit(X_train, y_train)
plt.figure(figsize=(18, 8))
plot_tree(tree, feature_names=data.feature_names, filled=True, rounded=True, fontsize=8)
plt.show()
```

```python
# Grid search
param_grid = {'max_depth': [10, 20, 30, None], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
dtree_reg = DecisionTreeRegressor(random_state=42)
grid_search = GridSearchCV(estimator=dtree_reg, param_grid=param_grid, cv=5, n_jobs=-1,
                           scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_dtree_reg = grid_search.best_estimator_
rmse = mean_squared_error(y_test, best_dtree_reg.predict(X_test)) ** 0.5
print(f"Best parameters: {grid_search.best_params_}")
print(f"Test RMSE: {rmse}")

# Random search
param_dist = {'max_depth': randint(1, 20), 'min_samples_split': randint(2, 20), 'min_samples_leaf': randint(1, 20)}
random_search = RandomizedSearchCV(DecisionTreeRegressor(random_state=42), param_distributions=param_dist,
                                   n_iter=50, cv=5, random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)
print(f"Best Parameters (Random Search): {random_search.best_params_}")
print(f"Best Score (Random Search): {random_search.best_score_}")
```

```python
# Bayesian optimisation  (pip install bayesian-optimization)
from bayes_opt import BayesianOptimization

def dtree_cv(max_depth, min_samples_split, min_samples_leaf):
    estimator = DecisionTreeRegressor(max_depth=int(max_depth), min_samples_split=int(min_samples_split),
                                      min_samples_leaf=int(min_samples_leaf), random_state=2)
    cval = cross_val_score(estimator, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
    return cval.mean()                       # optimizer maximises, so the negative MSE is maximised

param_bounds = {'max_depth': (1, 20), 'min_samples_split': (2, 20), 'min_samples_leaf': (1, 20)}
optimizer = BayesianOptimization(f=dtree_cv, pbounds=param_bounds, random_state=1)
optimizer.maximize(n_iter=25, init_points=5)

best_params_bayes = {k: int(v) for k, v in optimizer.max['params'].items()}
print(f"Best Parameters (Bayesian Optimization): {best_params_bayes}")
print(f"Best Score (Bayesian Optimization): {optimizer.max['target']}")
```

### 4d Support Vector Machine

```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

cancer = load_breast_cancer()
X = cancer.data[:, :2]                     # first two features so the boundary can be drawn in 2-D
y = cancer.target

svm = SVC(kernel="rbf", gamma=0.5, C=1.0)
svm.fit(X, y)
print("Training accuracy:", svm.score(X, y))

# also check on unseen data
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
print("Test accuracy    :", SVC(kernel="rbf", gamma=0.5, C=1.0).fit(Xtr, ytr).score(Xte, yte))

DecisionBoundaryDisplay.from_estimator(svm, X, response_method="predict", cmap=plt.cm.Spectral, alpha=0.8,
                                       xlabel=cancer.feature_names[0], ylabel=cancer.feature_names[1])
plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolors="k")
plt.show()
```

### 4e Random forest vs single decision tree

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

iris = datasets.load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# single decision tree
decision_tree = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
y_pred_tree = decision_tree.predict(X_test)
print("Decision Tree Performance:")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_tree))
print("\nClassification Report:\n", classification_report(y_test, y_pred_tree))
tree_acc = accuracy_score(y_test, y_pred_tree)
print("Accuracy Score:", tree_acc)

# random forest with different numbers of trees
n_trees = [5, 10, 50, 100]
accuracies = []
for n in n_trees:
    random_forest = RandomForestClassifier(n_estimators=n, random_state=42, max_features='sqrt')
    random_forest.fit(X_train, y_train)
    y_pred_forest = random_forest.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred_forest))
    print(f"\nRandom Forest with {n} trees - accuracy {accuracies[-1]:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_forest))

# feature sampling (max_features)
print("\nEffect of feature sampling (100 trees):")
for mf in ['sqrt', 'log2', None]:
    rf = RandomForestClassifier(n_estimators=100, max_features=mf, random_state=42).fit(X_train, y_train)
    print(f"  max_features={str(mf):<5} accuracy {rf.score(X_test, y_test):.4f}")

plt.figure(figsize=(10, 5))
plt.plot(n_trees, accuracies, marker='o', linestyle='-', color='b', label='Random Forest')
plt.axhline(y=tree_acc, color='r', linestyle='--', label='Decision Tree Accuracy')
plt.title('Random Forest Accuracy vs. Number of Trees')
plt.xlabel('Number of Trees'); plt.ylabel('Accuracy'); plt.xticks(n_trees); plt.grid(); plt.legend()
plt.show()
```

### 4f Gradient boosting – XGBoost with tuning and feature importance

```python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

path = 'Churn_Modelling.csv'
if os.path.exists(path):
    dataset = pd.read_csv(path)
else:                                         # generated sample with the same column order
    rng = np.random.default_rng(0)
    n = 2000
    age = rng.integers(18, 80, n)
    balance = rng.gamma(2, 40000, n) * (rng.random(n) > 0.3)
    exited = ((age > 50) | (rng.random(n) < 0.1)).astype(int)
    dataset = pd.DataFrame({
        'RowNumber': range(1, n + 1), 'CustomerId': rng.integers(10**7, 10**8, n),
        'Surname': rng.choice(['Smith', 'Rao', 'Khan'], n), 'CreditScore': rng.integers(350, 850, n),
        'Geography': rng.choice(['France', 'Spain', 'Germany'], n), 'Gender': rng.choice(['Male', 'Female'], n),
        'Age': age, 'Tenure': rng.integers(0, 11, n), 'Balance': balance,
        'NumOfProducts': rng.integers(1, 4, n), 'HasCrCard': rng.integers(0, 2, n),
        'IsActiveMember': rng.integers(0, 2, n), 'EstimatedSalary': rng.uniform(10000, 200000, n),
        'Exited': exited})
    print("Churn_Modelling.csv not found - generated sample data used")

X = dataset.iloc[:, 3:13].copy()              # CreditScore ... EstimatedSalary
y = dataset.iloc[:, 13].values                # Exited
X['Geography'] = X['Geography'].astype('category')
X['Gender'] = X['Gender'].astype('category')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
xgb_train = xgb.DMatrix(X_train, y_train, enable_categorical=True)
xgb_test = xgb.DMatrix(X_test, y_test, enable_categorical=True)

params = {'objective': 'binary:logistic', 'max_depth': 3, 'learning_rate': 0.1}
model = xgb.train(params=params, dtrain=xgb_train, num_boost_round=50)
probs = model.predict(xgb_test)                       # probabilities between 0 and 1
preds = (probs > 0.5).astype(int)                     # FIX: threshold at 0.5 (astype(int) alone gives all 0)
print('Accuracy of the model is:', accuracy_score(y_test, preds) * 100)
```

```python
# hyper-parameter tuning with GridSearchCV
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(
    xgb.XGBClassifier(tree_method='hist', enable_categorical=True, n_estimators=50, random_state=0),
    {'max_depth': [2, 3, 5], 'learning_rate': [0.05, 0.1, 0.3]}, cv=3, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)
print("Best parameters:", grid.best_params_)
print("Best CV accuracy:", round(grid.best_score_, 4))
print("Test accuracy   :", round(grid.score(X_test, y_test), 4))

# feature importance
xgb.plot_importance(model, importance_type='gain', max_num_features=10)
plt.title('XGBoost feature importance (gain)')
plt.show()
print(model.get_score(importance_type='gain'))
```

---

## Practical 5 – Generative models

### 5.1 Naive Bayes classifier

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred) * 100)
print("Confusion matrix:\n", metrics.confusion_matrix(y_test, y_pred))

# classify one new test sample
sample = [[5.1, 3.5, 1.4, 0.2]]
print("Test sample", sample, "->", iris.target_names[gnb.predict(sample)[0]],
      "| class probabilities:", gnb.predict_proba(sample).round(3))
```

### 5.2 Hidden Markov Model (hmmlearn)

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn import hmm

states = ["Sunny", "Rainy"]
n_states = len(states)
print('Number of hidden states :', n_states)
observations = ["Dry", "Wet"]
n_observations = len(observations)
print('Number of observations :', n_observations)

state_probability = np.array([0.6, 0.4])
print("State probability: ", state_probability)
transition_probability = np.array([[0.7, 0.3],
                                   [0.3, 0.7]])
print("\nTransition probability:\n", transition_probability)
emission_probability = np.array([[0.9, 0.1],
                                 [0.2, 0.8]])
print("\nEmission probability:\n", emission_probability)

model = hmm.CategoricalHMM(n_components=n_states)
model.startprob_ = state_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability

observations_sequence = np.array([0, 1, 0, 1, 0, 0]).reshape(-1, 1)
hidden_states = model.predict(observations_sequence)
print("Most likely hidden states:", hidden_states)

log_probability, hidden_states = model.decode(observations_sequence, lengths=len(observations_sequence),
                                              algorithm='viterbi')
print('Log Probability :', log_probability)
print("Most likely hidden states:", hidden_states)
print("As weather:", [states[i] for i in hidden_states])

sns.set_style("whitegrid")
plt.plot(hidden_states, '-o', label="Hidden State")
plt.xlabel('Time step'); plt.ylabel('Most Likely Hidden State'); plt.title("Sunny (0) or Rainy (1)")
plt.legend()
plt.show()
```

---

## Practical 6 – Probabilistic models

### 6.1a Bayesian linear regression with Pyro (prior and posterior)

```python
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoNormal
from pyro.optim import Adam
import matplotlib.pyplot as plt
import seaborn as sns

pyro.clear_param_store()
torch.manual_seed(0)
pyro.set_rng_seed(0)

X = torch.linspace(0, 10, 100)
true_slope, true_intercept = 2, 1
Y = true_intercept + true_slope * X + torch.randn(100)

def model(X, Y=None):
    slope = pyro.sample("slope", dist.Normal(0., 10.))              # priors
    intercept = pyro.sample("intercept", dist.Normal(0., 10.))
    sigma = pyro.sample("sigma", dist.HalfNormal(1.))
    mu = intercept + slope * X
    with pyro.plate("data", len(X)):
        pyro.sample("obs", dist.Normal(mu, sigma), obs=Y)           # likelihood

guide = AutoNormal(model)                                            # approximate posterior
svi = SVI(model, guide, Adam({"lr": 0.05}), loss=Trace_ELBO())
num_iterations = 2000
for i in range(num_iterations):
    loss = svi.step(X, Y)
    if (i + 1) % 500 == 0:
        print(f"Iteration {i + 1}/{num_iterations} - Loss: {loss:.2f}")
```

```python
sites = ("slope", "intercept", "sigma")
prior = Predictive(model, num_samples=1000, return_sites=sites)(X)                        # prior samples
posterior = Predictive(model, guide=guide, num_samples=1000, return_sites=sites)(X, Y)    # posterior samples

for name in sites:
    print(f"Estimated {name:<9}: {posterior[name].mean().item():.3f}")
print("(true slope = 2, true intercept = 1, true sigma = 1)")

fig, axs = plt.subplots(1, 3, figsize=(15, 4))
for ax, name in zip(axs, sites):
    sns.kdeplot(prior[name].detach().numpy().ravel(), fill=True, ax=ax, label='prior')
    sns.kdeplot(posterior[name].detach().numpy().ravel(), fill=True, ax=ax, label='posterior')
    ax.set_title(f"Prior vs posterior of {name}")
    ax.legend()
plt.tight_layout()
plt.show()
```

### 6.1b Bayesian linear regression with PyMC (MCMC)

```python
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

np.random.seed(0)
X = np.linspace(0, 10, 100)
true_slope, true_intercept = 2, 1
Y = true_intercept + true_slope * X + np.random.normal(0, 1, size=100)

with pm.Model() as model:
    slope = pm.Normal('slope', mu=0, sigma=10)
    intercept = pm.Normal('intercept', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=1)
    mu = intercept + slope * X
    Y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=Y)
    trace = pm.sample(2000, tune=1000, chains=2, random_seed=0)

print(az.summary(trace, var_names=['slope', 'intercept', 'sigma']))
az.plot_posterior(trace, var_names=['slope', 'intercept', 'sigma'])
plt.show()
```

### 6.2 Gaussian Mixture Model – density estimation and clustering

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.mixture import GaussianMixture

iris = datasets.load_iris()
X = iris.data[:, :2]
d = pd.DataFrame(X)

plt.scatter(d[0], d[1])
plt.title('Raw data')
plt.show()

gmm = GaussianMixture(n_components=3, random_state=0)
gmm.fit(d)
labels = gmm.predict(d)
d['labels'] = labels

d0, d1, d2 = d[d['labels'] == 0], d[d['labels'] == 1], d[d['labels'] == 2]
plt.scatter(d0[0], d0[1], c='r')
plt.scatter(d1[0], d1[1], c='yellow')
plt.scatter(d2[0], d2[1], c='g')
plt.title('GMM clusters')
plt.show()

print("Converged log-likelihood lower bound:", gmm.lower_bound_)
print("Iterations to converge:", gmm.n_iter_)
print("Cluster means:\n", gmm.means_.round(2))
print("Mixing weights:", gmm.weights_.round(3))
```

```python
# density estimation: probability density contours + choosing number of components with BIC
xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 200),
                     np.linspace(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5, 200))
density = np.exp(gmm.score_samples(np.c_[xx.ravel(), yy.ravel()])).reshape(xx.shape)
plt.contourf(xx, yy, density, levels=15, cmap='viridis')
plt.scatter(X[:, 0], X[:, 1], c='white', s=8)
plt.title('Density estimated by the GMM')
plt.show()

for k in range(1, 7):
    g = GaussianMixture(n_components=k, random_state=0).fit(X)
    print(f"components={k}  BIC={g.bic(X):.1f}")        # lowest BIC = best number of clusters
```

---

## Practical 7 – Model evaluation and hyper-parameter tuning

### 7.1 Cross-validation (stratified k-fold, k-fold)

```python
from statistics import mean, stdev
from sklearn import preprocessing, linear_model, datasets
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.pipeline import make_pipeline

cancer = datasets.load_breast_cancer()
x = cancer.data
y = cancer.target

scaler = preprocessing.MinMaxScaler()
x_scaled = scaler.fit_transform(x)
lr = linear_model.LogisticRegression()

# Stratified 10-fold: every fold keeps the same class ratio
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
lst_accu_stratified = []
for train_index, test_index in skf.split(x, y):
    x_train_fold, x_test_fold = x_scaled[train_index], x_scaled[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]
    lr.fit(x_train_fold, y_train_fold)
    lst_accu_stratified.append(lr.score(x_test_fold, y_test_fold))

print('List of possible accuracy:', lst_accu_stratified)
print('\nMaximum Accuracy That can be obtained from this model is:', max(lst_accu_stratified) * 100, '%')
print('\nMinimum Accuracy:', min(lst_accu_stratified) * 100, '%')
print('\nOverall Accuracy:', mean(lst_accu_stratified) * 100, '%')
print('\nStandard Deviation is:', stdev(lst_accu_stratified))
```

```python
# the same with cross_val_score (scaler inside a pipeline = no data leakage)
pipe = make_pipeline(preprocessing.MinMaxScaler(), linear_model.LogisticRegression())

kf_scores = cross_val_score(pipe, x, y, cv=KFold(n_splits=5, shuffle=True, random_state=1))
print("Plain 5-fold      :", kf_scores.round(3), "mean =", round(kf_scores.mean(), 4))

skf_scores = cross_val_score(pipe, x, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=1))
print("Stratified 5-fold :", skf_scores.round(3), "mean =", round(skf_scores.mean(), 4))
```

### 7.2 Grid search and randomized search

```python
from scipy.stats import randint
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Grid search: tries every combination
param_grid = {'max_depth': [3, 5, 7], 'min_samples_split': [2, 5, 10]}
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Grid search best params:", grid_search.best_params_)
print("Grid search best CV score:", round(grid_search.best_score_, 4))
print("Test accuracy:", grid_search.score(X_test, y_test))

# Randomized search: samples n_iter random combinations (faster for big spaces)
param_dist = {'max_depth': randint(1, 10), 'min_samples_split': randint(2, 20), 'min_samples_leaf': randint(1, 10)}
random_search = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), param_dist, n_iter=20, cv=5,
                                   random_state=42)
random_search.fit(X_train, y_train)
print("\nRandomized search best params:", random_search.best_params_)
print("Randomized search best CV score:", round(random_search.best_score_, 4))
print("Test accuracy:", random_search.score(X_test, y_test))
```

---

## Practical 8 – Bayesian learning (posterior of a Normal mean and sigma)

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
true_mu = 5
true_sigma = 2
data = np.random.normal(true_mu, true_sigma, size=100)
n = len(data)

# prior hyper-parameters
prior_mu_mean = 0
prior_mu_precision = 1          # variance = 1 / precision
prior_alpha = 2                 # Gamma(alpha, beta) prior on the precision tau = 1 / sigma^2
prior_beta = 2

# posterior of mu (Normal prior, sigma treated as known)
posterior_mu_precision = prior_mu_precision + n / true_sigma ** 2
posterior_mu_mean = (prior_mu_precision * prior_mu_mean + np.sum(data) / true_sigma ** 2) / posterior_mu_precision

# posterior of the precision tau (Gamma)
posterior_alpha = prior_alpha + n / 2
posterior_beta = prior_beta + np.sum((data - np.mean(data)) ** 2) / 2

# draw samples from the posteriors
posterior_mu = np.random.normal(posterior_mu_mean, 1 / np.sqrt(posterior_mu_precision), size=10000)
posterior_tau = np.random.gamma(posterior_alpha, 1 / posterior_beta, size=10000)   # numpy: scale = 1 / rate
posterior_sigma = 1 / np.sqrt(posterior_tau)                                       # convert precision -> sigma

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(posterior_mu, bins=30, density=True, color='skyblue', edgecolor='black')
plt.axvline(true_mu, color='red', ls='--', label='true value')
plt.title(r'Posterior distribution of $\mu$')
plt.xlabel(r'$\mu$'); plt.ylabel('Density'); plt.legend()

plt.subplot(1, 2, 2)
plt.hist(posterior_sigma, bins=30, density=True, color='lightgreen', edgecolor='black')
plt.axvline(true_sigma, color='red', ls='--', label='true value')
plt.title(r'Posterior distribution of $\sigma$')
plt.xlabel(r'$\sigma$'); plt.ylabel('Density'); plt.legend()
plt.tight_layout()
plt.show()

print("Mean of mu:", np.mean(posterior_mu))
print("Standard deviation of mu:", np.std(posterior_mu))
print("Mean of sigma:", np.mean(posterior_sigma))
print("Standard deviation of sigma:", np.std(posterior_sigma))
print("95% credible interval for mu:", np.percentile(posterior_mu, [2.5, 97.5]).round(3))
```

---

## Practical 9 – Deep generative model: GAN on CIFAR-10 (PyTorch)

Use a GPU in Colab (Runtime → Change runtime type → GPU). For a quick demo set `num_epochs = 2`.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", device)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])       # pixels -> [-1, 1] to match tanh output
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# hyper-parameters
latent_dim = 100
lr = 0.0002
beta1 = 0.5
beta2 = 0.999
num_epochs = 10
```

```python
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(1, (128, 8, 8)),
            nn.Upsample(scale_factor=2),                                   # 8 -> 16
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128, momentum=0.78),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),                                   # 16 -> 32
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64, momentum=0.78),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh())

    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.BatchNorm2d(64, momentum=0.82),
            nn.LeakyReLU(0.25),
            nn.Dropout(0.25),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128, momentum=0.82),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=0.8),
            nn.LeakyReLU(0.25),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(256 * 5 * 5, 1),
            nn.Sigmoid())                                                  # probability that the image is real

    def forward(self, img):
        return self.model(img)


generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)
adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))
print("Generator and discriminator created")
```

```python
for epoch in range(num_epochs):
    for i, batch in enumerate(dataloader):
        real_images = batch[0].to(device)
        valid = torch.ones(real_images.size(0), 1, device=device)           # label 1 = real
        fake = torch.zeros(real_images.size(0), 1, device=device)           # label 0 = fake

        # ---- train the discriminator ----
        optimizer_D.zero_grad()
        z = torch.randn(real_images.size(0), latent_dim, device=device)
        fake_images = generator(z)
        real_loss = adversarial_loss(discriminator(real_images), valid)
        fake_loss = adversarial_loss(discriminator(fake_images.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # ---- train the generator (try to fool the discriminator) ----
        optimizer_G.zero_grad()
        gen_images = generator(z)
        g_loss = adversarial_loss(discriminator(gen_images), valid)
        g_loss.backward()
        optimizer_G.step()

        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}] Batch {i + 1}/{len(dataloader)} "
                  f"Discriminator Loss: {d_loss.item():.4f} Generator Loss: {g_loss.item():.4f}")

    # show 16 generated images every 2 epochs and at the end
    if (epoch + 1) % 2 == 0 or epoch + 1 == num_epochs:
        with torch.no_grad():
            z = torch.randn(16, latent_dim, device=device)
            generated = generator(z).detach().cpu()
        grid = torchvision.utils.make_grid(generated, nrow=4, normalize=True)
        plt.imshow(np.transpose(grid, (1, 2, 0)))
        plt.axis("off")
        plt.title(f"Generated images after epoch {epoch + 1}")
        plt.show()
```

---

## Practical 10 – Deploy a model as an API (FastAPI)

**Step 1 – install**
```python
%pip install -q fastapi uvicorn requests scikit-learn
```

**Step 2 – write the API file** (`basic_app.py`)
```python
%%writefile basic_app.py
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB

app = FastAPI()                                   # FastAPI instance


# request body and type hints of each attribute
class request_body(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# train the model once when the app starts
iris = load_iris()
X = iris.data
Y = iris.target
clf = GaussianNB()
clf.fit(X, Y)


@app.get('/')
def main():
    return {'message': 'Welcome Everyone!'}


@app.get('/{name}')
def hello_name(name: str):
    return {'message': f'Welcome Everyone!, {name}'}


@app.post('/predict')
def predict(data: request_body):
    test_data = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    class_idx = clf.predict(test_data)[0]
    return {'class': str(iris.target_names[class_idx])}
```

**Step 3 – start the server and test it from the notebook**
```python
import subprocess, sys, time, requests

server = subprocess.Popen([sys.executable, "-m", "uvicorn", "basic_app:app", "--port", "8000"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
BASE = "http://127.0.0.1:8000"
for _ in range(30):                                   # wait until the server is up
    try:
        requests.get(BASE, timeout=1)
        break
    except requests.ConnectionError:
        time.sleep(0.5)

print("GET /        ->", requests.get(f"{BASE}/").json())
print("GET /Aisha   ->", requests.get(f"{BASE}/Aisha").json())
sample = {"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.2, "petal_width": 2.3}
print("POST /predict ->", requests.post(f"{BASE}/predict", json=sample).json())
setosa = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
print("POST /predict ->", requests.post(f"{BASE}/predict", json=setosa).json())

server.terminate()
server.wait()
```

**On a normal PC instead of the notebook (what the journal shows)** – save the Step 2 code (without the `%%writefile` line) as `basic_app.py`, then in CMD:
```
pip install fastapi uvicorn
uvicorn basic_app:app --reload
```
Open `http://127.0.0.1:8000/` for the welcome message and `http://127.0.0.1:8000/docs` for the Swagger page → POST `/predict` → *Try it out* → *Execute*.
