import os
import pandas as pd
train_dir = 'train_dataset'
test_dir = 'test_dataset'

X_train = pd.read_csv(os.path.join(train_dir, 'X_train_preprocessed.csv'))
X_test = pd.read_csv(os.path.join(test_dir, 'X_test_preprocessed.csv'))
y_train = pd.read_csv(os.path.join(train_dir, 'y_train.csv')).squeeze()  # .squeeze() to convert DataFrame to Series
y_test = pd.read_csv(os.path.join(test_dir, 'y_test.csv')).squeeze()  # .squeeze() to convert DataFrame to Series
print(X_train.columns)
