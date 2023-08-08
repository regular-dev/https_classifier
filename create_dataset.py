import numpy as np
import pickle
import sys
import dpkt
import datetime
from matplotlib import pyplot as plt
from dpkt.utils import mac_to_str, inet_to_str
from colorama import Fore, Back, Style
from datetime import datetime

_pcaps = [ "dump/Firefox_day1.pcap" , "dump/Google-Chrome_day2.pcap",
           "dump/Firefox_day3.pcap" , "dump/Google-Chrome_day4.pcap",
           "dump/Firefox_day5.pcap" , "dump/Google-Chrome_day6.pcap",
           "dump/Firefox_day7.pcap" , "dump/Google-Chrome_day8.pcap",
           "dump/Firefox_day9.pcap" , "dump/Google-Chrome_day10.pcap",
           "dump/Firefox_day11.pcap", "dump/Google-Chrome_day12.pcap"]
#_pcaps = [ "dump/test/Firefox_day2.pcap", "dump/test/Google-Chrome_day3.pcap" ] # test_net pcap
_input_sites_file = "https_websites_list.txt"
_crop_sec = 3.5
_out_file = "https_dataset.pickle"
_packet_limit = 1000

def check_with_ip(_assoc_ip, src_ip, dst_ip):
    for key, ip_arr in _assoc_ip.items():
        if src_ip in ip_arr and dst_ip in ip_arr:
            sr = 0.0 # 0 - recv from server, 1 - send to server

            if src_ip == ip_arr[0]:
                sr = 1.0

            return (key, sr)
    return None

def process_pcap(_file, _all_sites):
    assoc_ip = { }
    assoc_ts = { }
    assoc_start = { }
    data = { }

    for l in _all_sites:
        assoc_ip[l.strip()] = ['0.0.0.0', '0.0.0.0'] # src, dst
        data[l.strip()] = []

    with open(_file, 'rb') as f:
        pcap = dpkt.pcap.Reader(f)
        cnt = 0

        for ts, buf in pcap:
            eth = dpkt.ethernet.Ethernet(buf)

            if not isinstance(eth.data, dpkt.ip.IP):
                cnt = cnt + 1
                continue

            ip = eth.data

            ip_src_str = inet_to_str(ip.src).strip()
            ip_dst_str = inet_to_str(ip.dst).strip()

            if not isinstance(ip.data, dpkt.tcp.TCP):
                cnt = cnt + 1
                continue

            tcp = ip.data

            _len_tcp = len(tcp.data)

            try:
                tls = dpkt.ssl.TLS(tcp.data)
                if len(tls.records) < 1:
                    cnt = cnt + 1
                    continue

                tls_rec0 = tls.records[0]
                tls_h = dpkt.ssl.TLSHandshake(tls_rec0.data)

                if isinstance(tls_h.data, dpkt.ssl.TLSClientHello):
                    tls_cli_h = tls_h.data

                    srv_ext_id = -1
                    for idx, e in enumerate(tls_cli_h.extensions):
                        if e[0] == 0:
                            srv_ext_id = idx
                            break

                    if srv_ext_id == -1:
                        print("Couldn't find srv extension")
                        continue

                    tls_srv_name = tls_cli_h.extensions[srv_ext_id][1][5:].decode("utf-8")

                    #print("tls server name : {}".format(tls_srv_name))

                    if tls_srv_name in assoc_ip:
                        if tls_srv_name in assoc_ts.keys():
                            print("Duplicate : {}".format(tls_srv_name))
                            continue

                        assoc_ip[tls_srv_name] = [ip_src_str, ip_dst_str]
                        assoc_start[tls_srv_name] = ts
                        assoc_ts[tls_srv_name] = ts
                        print("Found : {}".format(tls_srv_name))
                        cnt = cnt + 1
                        continue

            except Exception as e:
                a = 0

            _site_ch = check_with_ip(assoc_ip, ip_src_str, ip_dst_str)

            if _site_ch is None:
                cnt += 1
                continue

            if ts - assoc_start[_site_ch[0]] > _crop_sec:
                print("overtime! continuing...")
                continue

            delta_ts = ts - assoc_ts[_site_ch[0]]
            assoc_ts[_site_ch[0]] = ts

            if len(data[_site_ch[0]]) > _packet_limit:
                print(Fore.RED + "over packet limit : {}".format(_site_ch[0]) + Style.RESET_ALL)
                continue

            data[_site_ch[0]].append([delta_ts, _site_ch[1], _len_tcp])
            cnt += 1
        return data

def process_files():
    all_data = {}

    _dt_start = datetime.now()

    with open(_input_sites_file, 'r') as f:
        lines = f.readlines()

        for l in lines:
            all_data[l.strip()] = []

    for f in _pcaps:
        print("Processing {}...".format(f))
        out = process_pcap(f, all_data.keys())

        for site, arr in out.items():
            if len(arr) != 0:
                all_data[site].append(arr)

    cnt_full = 0
    cnt_empty = 0
    empty_sites = []

    for key, it in all_data.items():
       if len(it) == 0:
           cnt_empty += 1
           empty_sites.append(key)
       else:
           cnt_full += 1
           print("Site {} has {} streams stored".format(key, len(it)))

    print("Count with data sites : {}".format(cnt_full))
    print("Count with empty sites : {}".format(cnt_empty))

    print("Empty sites : {}".format(empty_sites))

    _dt_end = datetime.now()

    print(Fore.CYAN + "Time elapsed : {} secs".format((_dt_end - _dt_start).total_seconds()) + Style.RESET_ALL)

    with open(_out_file, 'wb') as fo:
        print("Serializing data to {}".format(_out_file))
        pickle.dump(all_data, fo)


if __name__ == '__main__':
    process_files()
