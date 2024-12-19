from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from pickle import dump, load

filename = "./data/1_pima.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe =read_csv(filename, names=names)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=7)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

score = model.score(X_test, Y_test)

output_filename = 'finalized_model_with_score.sav'
with open(output_filename, 'wb') as file:
    dump({'model': model, 'score': score}, file)

with open(output_filename, 'rb') as file:
    loaded_data = load(file)

from pickle import load

output_filename = 'finalized_model_with_score.sav'
with open(output_filename, 'rb') as file:
    loaded_data = load(file)

loaded_model = loaded_data['model']
loaded_score = loaded_data['score']

print("Loaded Model Score", loaded_score)