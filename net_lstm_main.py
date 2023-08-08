import numpy as np
import pickle
import tensorflow as tf
import datetime
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, \
                                    Conv1D, Reshape, Flatten, ConvLSTM2D
from colorama import Fore, Back, Style
from datetime import datetime
import sys
import math
from random import shuffle
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model

__dataset_file = 'https_dataset.pickle'
__test_dataset = 'test_https_dataset.pickle'
__split_packets = 51 # 51 IMPORTANT : IF MAKE CHANGES, CHANGE RESHAPING ON TRAINING LOOP
__size_of_packet = 3
__epochs = 1
__batch_size = 32

__max_pkt_len = 1300.0

def complete_arr(_arr, _size):
    if len(_arr) > _size * __size_of_packet:
        print(Fore.RED + "Length of arr is {}, which is more than {}".format(len(_arr), __split_packets * __size_of_packet) + Style.RESET_ALL)
        return None

    while len(_arr) < _size * __size_of_packet:
        _arr.append(0.0)

    return _arr

# kinda StandardScaler
def do_zscore(_arr):
    if len(_arr) == 0:
        return []
    if len(_arr) == 1:
        return _arr

    mean = sum(_arr) / (len(_arr))
    differences = [(value - mean)**2 for value in _arr]
    sum_of_differences = sum(differences)
    standard_deviation = (sum_of_differences / (len(_arr) - 1)) ** 0.5

    zscores = [(value - mean) / standard_deviation for value in _arr]
    return zscores

def ones_list(n):
    listofones = [1.0] * n
    return listofones

def do_minmax(_arr):
    if len(_arr) == 0:
        return []
    if len(_arr) == 1:
        return [1.0]

    _arr_min = min(_arr)
    _arr_max = max(_arr)

    if _arr_max - _arr_min == 0:
        return ones_list(len(_arr))

    _minmax = [(value - _arr_min) / (_arr_max - _arr_min) for value in _arr]
    return _minmax

def do_standartization_group(_x_groups, _zscores):
    _p_cnt = 0
    for (_g_idx, g) in enumerate(_x_groups):
        for (_idx_val, _val) in enumerate(g):
            if _idx_val % __size_of_packet == 0:
                if _p_cnt < len(_zscores):
                    _x_groups[_g_idx][_idx_val] = _zscores[_p_cnt]
                    _p_cnt += 1


def prepare_data(data_path, _do_shuffle):
    with open(data_path, 'rb') as fi:
        data_imported = pickle.load(fi)

        cnt_with_data = 0

        for idx, (key, val) in enumerate(data_imported.items()):
            if len(val) == 0:
                print(Fore.RED + "{} Site {} has no data".format(idx, key) + Style.RESET_ALL)
            else:
                cnt_with_data += 1

        print(Fore.CYAN + "Sites with data : {}".format(cnt_with_data) + Style.RESET_ALL)

        dataset = []

        for idx, (key, arr) in enumerate(data_imported.items()):

            for inner_arr in arr:
                x_rows = [] # single site capture
                x_temp = []
                _len_inner_arr = len(inner_arr)
                pkt_cnt = 0
                zscore_ts_helper = []

                for idx_pkt, each_packet in enumerate(inner_arr):
                    if idx_pkt == _len_inner_arr - 1:
                        x_temp = complete_arr(x_temp, __split_packets)
                        x_rows.append(x_temp)
                        continue

                    if each_packet[2] == 0:
                        continue

                    zscore_ts_helper.append(each_packet[0])

                    x_temp.append(each_packet[0]) # delta ts
                    x_temp.append(each_packet[1]) # recv
                    x_temp.append(each_packet[2] / __max_pkt_len) # length

                    pkt_cnt += 1

                    if pkt_cnt % __split_packets == 0 and pkt_cnt != 0:
                        x_rows.append(x_temp)
                        x_temp = []

                ts_zscores = do_zscore(zscore_ts_helper)
                ts_minmax = do_minmax(ts_zscores)
                do_standartization_group(x_rows, ts_minmax)

                dataset.append([x_rows, idx])

        print("dataset size : {}".format(len(dataset)))
        if _do_shuffle == True:
            shuffle(dataset)
        return dataset

def conv_net_constructor():
    model = Sequential()

    model.add(ConvLSTM2D(filters=4, kernel_size=5, padding='same', activation='relu', return_sequences=True))
    model.add(ConvLSTM2D(filters=8, kernel_size=4, padding='same', activation='relu', return_sequences=True))
    model.add(ConvLSTM2D(filters=16, kernel_size=3, padding='same', activation='relu', return_sequences=True))
    model.add(ConvLSTM2D(filters=24, kernel_size=2, padding='same', activation='relu', return_sequences=False))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(776, activation='softmax'))

    return model

def lstm_net_constructor():
    model = Sequential()

    model.add(LSTM(500, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(420, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(776, activation='softmax'))

    return model

def fit_net(dataset, is_conv):
    _bench_start = datetime.now()

    if is_conv:
        model = conv_net_constructor()
    else:
        model = lstm_net_constructor()

    opt = tf.keras.optimizers.Adam(learning_rate=3e-4)
    train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    for epoch_idx in range(__epochs):
        loss = 0.0

        print(Fore.BLUE + "Epoch {}".format(epoch_idx) + Style.RESET_ALL)

        __each_n = __batch_size*15

        list_grads = []

        _len_dataset = len(dataset)

        collect_x_arr = [] # for batch training
        collect_y_arr = [] # for batch training

        for idx, x in enumerate(dataset):
            x_arr = np.expand_dims(np.array(x[0]), axis = 0)
            y_arr = np.array([dataset[idx][1]])

            if is_conv:
                x_arr = np.reshape(x_arr, (1, x_arr.shape[1], 1, 17, 9)) # for ConvLSTM2D

            collect_x_arr.append(x_arr)
            collect_y_arr.append(y_arr)

            if len(collect_x_arr) == __batch_size:
                current_loss = 0.0
                with tf.GradientTape() as tape:
                    for b_i in range(len(collect_x_arr)):
                        logits = model(collect_x_arr[b_i], training=True)
                        current_loss += loss_fn(y_true=collect_y_arr[b_i], y_pred=logits)
                        train_acc_metric.update_state(collect_y_arr[b_i], logits)
                        model.reset_states()

                    grad = tape.gradient(current_loss / __batch_size, model.trainable_weights)

                    opt.apply_gradients(zip(grad, model.trainable_weights))

                collect_x_arr = []
                collect_y_arr = []
                loss += current_loss

            if idx % __each_n == 0:
                print(Fore.BLUE + "Done : {:.3f} | Loss : {:.4f}".format(float(idx) / _len_dataset, loss / ( (idx/__batch_size) + 1e-7)) + Style.RESET_ALL)
                train_acc = train_acc_metric.result()
                print("Training acc over epoch: {:.4f}".format(float(train_acc)))

        print(Fore.GREEN + "Epoch {} loss : {:.4f}".format(epoch_idx, loss / ( (idx/__batch_size) + 1e-7)) + Style.RESET_ALL)

        f_stats.write("{:.3f},{:.3f}\n".format(loss / (idx/(__batch_size + 1e-8)), train_acc_metric.result()))

        train_acc_metric.reset_states()
        loss = 0.0

    _bench_end = datetime.now()

    print(Fore.CYAN + "Time elapsed : {} secs".format((_bench_end - _bench_start).total_seconds()) + Style.RESET_ALL)

    model.save("model.state")
    f_stats.close()

def test(_is_conv_test):
    model = load_model("model.state")
    dataset = prepare_data(__test_dataset, False)
    acc_cnt = 0
    acc_cnt_ff = 0
    acc_cnt_chr = 0
    bench_sum = 0

    for idx, x in enumerate(dataset):
        x_arr = np.expand_dims(np.array(x[0]), axis = 0)
        y_arr = dataset[idx][1]

        if _is_conv_test:
            x_arr = np.reshape(x_arr, (1, x_arr.shape[1], 1, 17, 9)) # reshaping for ConvLSTM2D

        _bench_start = datetime.now()
        model_out = model(x_arr, training=False)
        _bench_end = datetime.now()

        bench_sum += (_bench_end - _bench_start).total_seconds()

        model_y = np.argmax(model_out)

        if y_arr == model_y:
            acc_cnt += 1

            if idx >= 736:
                acc_cnt_chr += 1
            else:
                acc_cnt_ff +=1

    ff_accuracy = acc_cnt_ff / 736
    chr_accuracy = acc_cnt_chr / 736
    final_accuracy = acc_cnt / len(dataset)

    print("Chrome accuracy : {:.3f}. Recognized sites : {} / {}".format(chr_accuracy, acc_cnt_chr, 736))
    print("Firefox accuracy : {:.3f}. Recognized sites : {} / {}".format(ff_accuracy, acc_cnt_ff, 736))
    print("Final accuracy : {:.3f}".format(final_accuracy))
    print("Average time per site : {:.4f} milliseconds".format( (bench_sum / len(dataset)) * 1000.0 ))

if __name__ == '__main__':
    action = sys.argv[1]
    # change this to True if you want to use Conv2DLSTM instead of LSTM
    cfg_is_conv = False

    if action == "train":
        dataset = prepare_data(__dataset_file, cfg_is_conv)
        fit_net(dataset, cfg_is_conv)
    if action == "test":
        test(cfg_is_conv)
