# HTTPS Classifier
Keras LSTM & Conv2DLSTM for https traffic flows classification

# Usage
 - Install requirements.
 - Upload pcap files from https://betternet.lhs.inria.fr/datasets/https/dumps.html
 - Edit create_dataset.py list of pcap files
 - Launch ```python create_dataset.py``` to create serialized pickle dataset
 - Launch ```python net_lstm_main.py train``` to train net
 - Launch ```python net_lstm_main.py test``` to test net

