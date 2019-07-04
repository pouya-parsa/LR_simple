import tensorflow as tf
import numpy as np
import requests
import pandas as pd
from pandas import DataFrame as DF, Series

data = pd.read_csv("~/datasets/titanic/train.csv")

data.drop(["PassengerId", "Name", "Ticket"], axis=1, inplace=True)

data.to_csv("data.csv", index=False)

del data
import gc
gc.collect()


data = pd.read_csv("data.csv")
# print(data.isnull().sum())

data.fillna({
    'Age': -1,
    'Cabin': 'Unk',
    'Embarked': 'Unk',
    'Fare': -1
}, inplace=True)

# convert sex binary
data.loc[:, 'Sex'] = (data.Sex == 'female').astype(int)

Xtr = data.loc[:, ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].sample(frac=0.75)
Xts = data[~data.index.isin(Xtr.index)].loc[:, ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]

Ytr = pd.get_dummies(data[data.index.isin(Xtr.index)].Survived).values
Yts = pd.get_dummies(data[~data.index.isin(Xtr.index)].Survived).values


num_features = Xtr.shape[1]
num_classes = 2

X = tf.placeholder('float', [None, num_features])
Y = tf.placeholder('float', [None, num_classes])

# W - weights array
W = tf.Variable(tf.zeros([num_features, num_classes]))

# B - Bias array
B = tf.Variable(tf.zeros(num_classes))

# define the logistic model
# y=wx+b as argument of softmax
yhat = tf.matmul(X, W) + B

# define a loss function
loss_fn = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=yhat, labels=Y))

# define optimizer and minimize on loss_fn
opt = tf.train.AdamOptimizer(0.01).minimize(loss_fn)

# create session
sess = tf.Session()

# init vars
init = tf.initialize_all_variables()
sess.run(init)

num_epochs = 10

# loop over num_epochs and run optimization step on
# full data each time
for i in range(num_epochs):
    sess.run(opt, feed_dict={X: Xtr, Y: Ytr})
    # yhat1 = sess.run([loss_fn], feed_dict={X: Xtr, Y: Ytr})
    # print(yhat1)

# accuracy function
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(yhat, 1), tf.argmax(Y, 1)), 'float'))
# accuracy_x = tf.cast(tf.equal(tf.argmax(yhat, 1), tf.argmax(Y, 1)), 'float')

# get the test accuracy
accuracy_value = sess.run(accuracy, feed_dict={X: Xts, Y: Yts})

# print(accuracy_x)
print(accuracy_value)
