import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay


dataset_imported = []
dataset = []

def one_hot_encode(_id_class, _num_classes):
    a = np.zeros((1, _num_classes))
    a[0, _id_class] = 1
    # np.put(a, _id_class, 1)
    return a

with open('data.bin', 'rb') as fi:
    dataset_imported = pickle.load(fi)

for arr, lbl in dataset_imported:
    arr_np = np.zeros((1, len(arr), 3))

    print(arr_np.shape)

    i = 0
    for ts_delta, _len, sr in arr:
        arr_np[0, i, 0] = ts_delta
        arr_np[0, i, 1] = _len
        arr_np[0, i, 2] = sr
        i += 1

    # lbl_np = one_hot_encode(lbl, 4)
    lbl_np = np.array([lbl])

    dataset.append((arr_np, lbl_np))

arr1_x = np.array([[ [0.0, 5.0, 0.0], [10.0, 0.0, 0.0] ]] )
arr1_y = np.array([[1.0, 0.0]])

arr2_x = np.array([[ [10.0, 0.0, 0.0], [0.0, 5.0, 0.0] ]] )
arr2_y = np.array([0.0])

# activity_regularizer=regularizers.L2(1e-2) with relu activation

# model

model = Sequential()

model.add(LSTM(128, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(64, activation='tanh'))
model.add(Dropout(0.3))

model.add(Dense(64, activation=LeakyReLU(alpha=0.01))))
model.add(Dropout(0.3))

model.add(Dense(4, activation='softmax'))


lr_decay = ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=70,
    decay_rate=0.7)
opt = tf.keras.optimizers.RMSprop(learning_rate=lr_decay)

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# model.compile(loss='mean_squared_error',
#               optimizer=opt)

#model.fit(x_train, y_train, epochs=3, validation_data=(x_test,y_test))

for i_cnt in range(100):
    loss = 0.0
    for arr_np, lbl_np in dataset:
        loss += model.train_on_batch(arr_np, lbl_np)[0]
        model.reset_states()

    print("step : {} , loss : {}".format(i_cnt, loss))
    if loss < 0.01:
        break

i_cnt = 0
for arr_np, lbl_np in dataset:
    predict1 = model.predict(arr_np)
    print("predict {} is {}".format(i_cnt, predict1))
    i_cnt += 1

exit(0)

predict1 = model.predict(arr1_x)
print("predict arr1 : {}".format(predict1))

predict2 = model.predict(arr2_x)
print("predict arr2 : {}".format(predict2))

# mention: model.reset_states method after each batch train
