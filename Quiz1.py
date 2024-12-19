import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

filename = "./data/1_pima.csv"

column_names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

data = pd.read_csv(filename, names=column_names)

x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

scaler = MinMaxScaler()
x_scaler = scaler.fit_transform(x)

model = DecisionTreeClassifier(max_depth=1000, min_samples_split=60, min_samples_leaf=5)

kfold = KFold(n_splits=10, random_state=5, shuffle=True)
results = cross_val_score(model, x_scaler, y, cv=kfold, scoring='accuracy')

print(results.mean())