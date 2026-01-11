---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
import pandas as pd
import numpy as np
```

```python
car_data = pd.read_csv("data.csv")
```

```python
car_data.head()
```

```python
car_data.isnull().sum()
```

```python
car_data.info()
```

```python
%pip install seaborn
import matplotlib.pyplot as plt
import seaborn as sns
```

```python
target = "MSRP"
```

```python
car_data["Age"] = car_data["Year"].max() - car_data["Year"]
base = ["Age", "Engine HP", "Engine Cylinders", "Number of Doors", "highway MPG", "city mpg", "Popularity"]
```

```python
car_data['Transmission Type'].value_counts()
```

```python
for v in ['AUTOMATIC', 'MANUAL', 'AUTOMATED_MANUAL']:
    feature = f"is_transmission_{v}"
    car_data[feature] = (car_data['Transmission Type'] == v).astype(int)
    base.append(feature)
```

```python
base.append(target)
```

```python
car_data[base].sample(3)
```

```python
price_mean = np.mean(car_data[target])
np.log1p(price_mean)
```

```python
sns.histplot(car_data[target])
```

```python
# log plot
sns.histplot(np.log1p(car_data[target]))
```

```python
data = car_data[base]
data.head()
```

```python
np.random.seed(123)
n = len(data)
idx = np.arange(n)
np.random.shuffle(idx)
data_shuffled = data.iloc[idx]
```

```python
n_test = int(0.2*n)
n_val = int(0.2*n)
n_train = n - (n_val+n_test)
n_test, n_val, n_train
```

```python
train_data = data_shuffled[:n_train]
val_data = data_shuffled[n_train:n_train+n_val]
test_data = data_shuffled[n_train+n_val:]
```

```python
x_train = train_data.drop(columns=["MSRP"])
y_train = train_data["MSRP"]
x_val = val_data.drop(columns=["MSRP"])
y_val = val_data["MSRP"]
x_test = test_data.drop(columns=["MSRP"])
y_test = test_data["MSRP"]
```

```python
x_train.head()
```

```python
y_train.head()
```

```python
x_train_mat = x_train.values
x_val_mat = x_val.values
x_test_mat = x_test.values
```

```python
def train_linear_regression(X, y):
    # Add the 1 feature value for the bias term in our feature matrix
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    # Calculate XTX (Gram matrix)
    XTX = X.T.dot(X)
    # Calculate the inverse of the Gram matrix
    XTX_inv = np.linalg.inv(XTX)
    # Extract w
    w = XTX_inv.dot(X.T).dot(y)
    
    return w[0], w[1:]
```

```python
w0, w = train_linear_regression(x_train_mat, y_train)
y_pred_val = w0 + x_val_mat.dot(w)
```

```python
def rmse(y, y_pred):
    error = y_pred - y
    # Mean squared error (MSE)
    mse = (error ** 2).mean()
    # Return the root of the mean squared error
    return np.sqrt(mse)
```

```python

```
