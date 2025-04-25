import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print(train.shape)
print(train.columns)
print(train['label'].value_counts())
train.describe()
