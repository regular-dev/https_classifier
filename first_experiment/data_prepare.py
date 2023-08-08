import numpy as np
import pickle
import sys
import dpkt
import datetime
from matplotlib import pyplot as plt
from dpkt.utils import mac_to_str, inet_to_str


only_tls = 0
show_plot_len = False # change this variable to plot lengths of packets
show_plot_ts = False # change this variable to plot timestamps of packets
            # habr
path_arr = [("dataset/habr/1.pcap", "178.248.237.68", 0),
           ("dataset/habr/2.pcap", "178.248.237.68", 0),
            # vk
           ("dataset/vk/1.pcap", "87.240.132.78", 1),
           ("dataset/vk/2.pcap", "87.240.132.78", 1),
            # ya
           ("dataset/ya/1.pcap", "87.250.250.242", 2),
           ("dataset/ya/2.pcap", "87.250.250.242", 2),
            # youtube
           ("dataset/youtube/1.pcap", "209.85.233.198", 3),
           ("dataset/youtube/2.pcap", "173.194.222.93", 3)
            ]

dataset_arr = []

def normalize_len_minmax(dataset):
    el_min = 999999.0
    el_max = -999999.0

    # find min/max
    for arr, lbl in dataset:
        for ts_delta, _len, sr in arr:
            if _len > el_max:
                el_max = _len
            if _len < el_min:
                el_min = _len

    delta_el = el_max - el_min

    for arr, lbl in dataset:
        i = 0
        for ts_delta, _len, sr in arr:
            arr[i] = (ts_delta, (_len - el_min) / delta_el, sr)
            i += 1

def seq_int_arr(num):
    arr = np.zeros(num)
    for i in range(num):
        arr[i] = i

    return arr

def normalize_ts_zscore_global(dataset):
    mean = 0.0
    sum = 0.0
    all_len = 0

    for arr, lbl in dataset:
        for ts_delta, _, _ in arr:
            sum += ts_delta
            all_len += 1

    # compute mean
    mean = sum / all_len

    differences = []

    for arr, lbl in dataset:
        for ts_delta, _, _ in arr:
            differences.append((ts_delta - mean)**2)

    sum_of_differences = np.sum(differences)
    standard_deviation = (sum_of_differences / (all_len - 1)) ** 0.5

    # zscore compute finally
    for arr, lbl in dataset:
        ts_delta_arr = []
        for ts_delta, _len, sr in arr:
            arr[i] = ((ts_delta - mean) / standard_deviation, _len , sr)
            ts_delta_arr.append((ts_delta - mean) / standard_deviation)

        if show_plot_ts:
            seq_arr = seq_int_arr(len(arr))
            plt.bar(seq_arr, ts_delta_arr)
            plt.show()

def normalize_ts_zscore(dataset):
    for arr, lbl in dataset:
        _sum = 0.0

        for ts_delta, _, _ in arr:
            _sum += ts_delta

        mean = _sum / len(arr)

        differences = []

        for ts_delta,_,_ in arr:
            differences.append((ts_delta - mean) ** 2)

        sum_of_differences = np.array(differences).sum()
        standard_deviation = (sum_of_differences / (len(arr) - 1)) ** 0.5

        ts_delta_arr = []
        ts_prev_arr = []

        for ts_delta, _len, sr in arr:
            ts_prev_arr.append(ts_delta)
            arr[i] = ((ts_delta - mean) / standard_deviation, _len, sr)
            ts_delta_arr.append((ts_delta - mean) / standard_deviation)

        if show_plot_ts:
            seq_arr = seq_int_arr(len(arr))
            plt.bar(seq_arr, ts_prev_arr, color='green')
            plt.bar(seq_arr, ts_delta_arr, color='red', alpha = 0.3)
            plt.hist(ts_delta_arr)
            plt.show()


def write_pkt_data(pcap, focus_ip, output, label):
    i = 0
    j = 0
    prev_ts = 0
    local_arr = []

    for ts, buf in pcap:
        eth = dpkt.ethernet.Ethernet(buf)

        if not isinstance(eth.data, dpkt.ip.IP):
            #print('Non IP Packet type not supported %s\n' % eth.data.__class__.__name__)
            continue

        ip = eth.data

        if not isinstance(ip.data, dpkt.tcp.TCP):
            #print("Not TCP packet, but : {}".format(ip.data.__class__.__name__))
            continue

        tcp = ip.data

        len_tcp = len(tcp.data)

        sr = 'send'
        sr_num = 1

        ip_dst_str = inet_to_str(ip.dst)
        ip_src_str = inet_to_str(ip.src)

        if ip_dst_str != focus_ip and ip_src_str != focus_ip:
            continue

        if tcp.flags & dpkt.tcp.TH_SYN:
            continue

        if ip_dst_str == "192.168.152.228":
            sr_num = 0
            sr = "recv"

        cur_ts_delta = 0.0

        if i == 0:
            prev_ts = ts
            cur_ts_delta = 0.0
        else:
            cur_ts_delta = ts - prev_ts
            prev_ts = ts

        i += 1

        if only_tls == 0:
            np_arr = np.array([cur_ts_delta, len_tcp, sr_num])
            local_arr.append(np_arr)
            continue

        try:
            tls = dpkt.ssl.TLS(tcp.data)
            np_arr = np.array([cur_ts_delta, tls.len, sr_num])

            local_arr.append(np_arr)
        except:
            # print("Could'nt parse TLS")
            continue

    dataset_arr.append((local_arr, label))

    print('Packets count : %i \n' % i)

def create_dataset(output):
    for (input, focus_ip, label) in path_arr:
        print("Reading from {}...".format(input))

        with open(input, 'rb') as f:
            pcap = dpkt.pcap.Reader(f)
            write_pkt_data(pcap, focus_ip, output, label)

    print("Normalizing len with minmax...")
    normalize_len_minmax(dataset_arr)
    print("Normalizing timestamp delta with zscore...")
    normalize_ts_zscore(dataset_arr)

    print("Writing datset to file {}...".format(output))
    with open(output, 'wb') as fo:
        pickle.dump(dataset_arr, fo)


if __name__ == '__main__':
    arg_len = len(sys.argv)

    print("arg num : {}".format(arg_len))

    if arg_len != 2:
        print("wrong number of arguments ./data_prepare.py <output_file>")
        exit(1)

    create_dataset(sys.argv[1])
