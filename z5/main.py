import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Classifying banknotes as real or fake.
"""
print("<----->\nBanknote\n<----->")
banknote_data = pd.read_csv('banknotes.csv', names=['Variance', 'Skewness', 'Kurtosis', 'Entropy', 'Class'])

b_train_dataset = banknote_data.sample(frac=0.8, random_state=200)
b_test_dataset = banknote_data.drop(b_train_dataset.index)

b_train_data = b_train_dataset.copy()
b_train_label = b_train_data.pop('Class')
b_train_label = np.array(b_train_label)

b_test_data = b_test_dataset.copy()
b_test_label = b_test_data.pop('Class')
b_test_label = np.array(b_test_label)

banknote_model = tf.keras.Sequential([tf.keras.layers.Dense(64, activation='relu'),
                                      tf.keras.layers.Dense(1)
                                      ])
banknote_model.compile(optimizer='adam',
                       loss='mean_squared_error',
                       metrics=['accuracy']
                       )
banknote_model.fit(b_train_data, b_train_label, epochs=5)

"""
Classifying 3 different animals.
"""
print("<----->\nCifar10 Selected Animals\n<----->")
cifar10_data = tf.keras.datasets.cifar10
(c_train_data, c_train_label), (c_test_data, c_test_label) = cifar10_data.load_data()
c_train_data = c_train_data / 255.0
c_test_data = c_test_data / 255.0

train_cat_index = np.where(c_train_label.reshape(-1) == 3)
train_cat_data = c_train_data[train_cat_index]
train_cat_label = c_train_label[train_cat_index]

train_frog_index = np.where(c_train_label.reshape(-1) == 6)
train_frog_data = c_train_data[train_frog_index]
train_frog_label = c_train_label[train_frog_index]

train_horse_index = np.where(c_train_label.reshape(-1) == 7)
train_horse_data = c_train_data[train_horse_index]
train_horse_label = c_train_label[train_horse_index]

cifar3_train_data = np.concatenate((train_cat_data, train_frog_data, train_horse_data))
cifar3_train_label = np.concatenate((train_cat_label, train_frog_label, train_horse_label)).reshape(-1, 1)
cifar3_train_label[cifar3_train_label == 3] = 0
cifar3_train_label[cifar3_train_label == 6] = 1
cifar3_train_label[cifar3_train_label == 7] = 2

test_cat_index = np.where(c_test_label.reshape(-1) == 3)
test_cat_data = c_test_data[test_cat_index]
test_cat_label = c_test_label[test_cat_index]

test_frog_index = np.where(c_test_label.reshape(-1) == 6)
test_frog_data = c_test_data[test_frog_index]
test_frog_label = c_test_label[test_frog_index]

test_horse_index = np.where(c_test_label.reshape(-1) == 7)
test_horse_data = c_test_data[test_horse_index]
test_horse_label = c_test_label[test_horse_index]

cifar3_test_data = np.concatenate((test_cat_data, test_frog_data, test_horse_data))
cifar3_test_label = np.concatenate((test_cat_label, test_frog_label, test_horse_label)).reshape(-1, 1)
cifar3_test_label[cifar3_test_label == 3] = 0
cifar3_test_label[cifar3_test_label == 6] = 1
cifar3_test_label[cifar3_test_label == 7] = 2

cifar3_model = tf.keras.models.Sequential(
    [tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
     tf.keras.layers.MaxPooling2D((2, 2)),
     tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
     tf.keras.layers.MaxPooling2D((2, 2)),
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(64, activation='relu'),
     tf.keras.layers.Dense(3, activation='softmax')
     ])
cifar3_model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy']
                     )
cifar3_model.fit(cifar3_train_data, cifar3_train_label, epochs=10)

"""
Classifying 10 different types of clothes.
"""
print("<----->\nFashion Mnist\n<----->")
fashion_data = tf.keras.datasets.fashion_mnist
(f_train_data, f_train_label), (f_test_data, f_test_label) = fashion_data.load_data()
f_train_data = f_train_data / 255.0
f_test_data = f_test_data / 255.0

fashion_model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
                                     tf.keras.layers.Dense(128, activation='relu'),
                                     tf.keras.layers.Dense(10, activation='softmax')
                                     ])
fashion_model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy']
                      )
fashion_model.fit(f_train_data, f_train_label, epochs=10)

"""
Classifying age of abalones.
"""
print("<----->\nAbalone\n<----->")
abalone_data = pd.read_csv('abalones.csv',
                           names=['Length', 'Diameter', 'Height', 'Whole Weight', 'Shucked Weight', 'Viscera Weight',
                                  'Shell Weight', 'Age'])

a_train_dataset = abalone_data.sample(frac=0.8, random_state=200)
a_test_dataset = abalone_data.drop(a_train_dataset.index)

a_train_data = a_train_dataset.copy()
a_train_label = a_train_data.pop('Age')
a_train_label = np.array(a_train_label)

a_test_data = a_test_dataset.copy()
a_test_label = a_test_data.pop('Age')
a_test_label = np.array(a_test_label)

normalize = tf.keras.layers.Normalization()
normalize.adapt(a_train_data)
normalize.adapt(a_test_data)

abalone_model = tf.keras.Sequential([normalize,
                                     tf.keras.layers.Dense(128, activation='relu'),
                                     tf.keras.layers.Dense(64, activation='relu'),
                                     tf.keras.layers.Dense(28, activation='softmax')
                                     ])
abalone_model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy']
                      )
abalone_model.fit(a_train_data, a_train_label, epochs=10)

print(f"Models accuracy"
      f"\nBanknote: {round(banknote_model.evaluate(b_test_data, b_test_label)[1] * 100, 2)}%"
      f"\nCifar3: {round(cifar3_model.evaluate(cifar3_test_data, cifar3_test_label)[1] * 100, 2)}%"
      f"\nFashion: {round(fashion_model.evaluate(f_test_data, f_test_label)[1] * 100, 2)}%"
      f"\nAbalone: {round(abalone_model.evaluate(a_test_data, a_test_label)[1] * 100, 2)}%")
