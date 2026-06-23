import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
import wfdb
import ast
from scipy.signal import resample
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn import preprocessing


# DATA PROCESSING STUFF
def load_dataset(path, sampling_rate, release=False):
    X, Y = None, None  # 初始化变量

    # 检查路径是否包含 PTB-XL 数据集文件
    if os.path.exists(os.path.join(path, 'raw/ptbxl_database.csv')):
        # 加载 PTB-XL 数据集
        Y = pd.read_csv(os.path.join(path, 'raw/ptbxl_database.csv'), index_col='ecg_id')
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
        X = load_raw_data_ptbxl(Y, sampling_rate, path)

    # 检查路径是否包含 CPSC 数据集文件
    elif os.path.exists(os.path.join(path, 'cpsc_database.csv')):
        # 加载 CPSC 数据集
        Y = pd.read_csv(os.path.join(path, 'cpsc_database.csv'), index_col='ecg_id')
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
        X = load_raw_data_cpsc(Y, sampling_rate, path)

    else:
        raise ValueError(f"未知的数据集路径: {path}。路径中应包含 PTB-XL 或 CPSC 数据集文件。")

    if X is None or Y is None:
        raise ValueError("数据集加载失败，请检查路径和文件是否存在。")

    return X, Y


def load_raw_data_cpsc(df, sampling_rate, path):
    if sampling_rate == 100:
        if os.path.exists(os.path.join(path, 'raw100.npy')):
            data = np.load(os.path.join(path, 'raw100.npy'), allow_pickle=True)
        else:
            data = [wfdb.rdsamp(os.path.join(path, 'records100', str(f))) for f in tqdm(df.index)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(os.path.join(path, 'raw100.npy'), 'wb'), protocol=4)
    elif sampling_rate == 500:
        if os.path.exists(os.path.join(path, 'raw500.npy')):
            data = np.load(os.path.join(path, 'raw500.npy'), allow_pickle=True)
        else:
            data = [wfdb.rdsamp(os.path.join(path, 'records500', str(f))) for f in tqdm(df.index)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(os.path.join(path, 'raw500.npy'), 'wb'), protocol=4)
    return data


def sliding_window_split(signal, window_size_samples, stride_samples):
    segments = []
    n_samples = len(signal)
    start = 0
    while start + window_size_samples <= n_samples:
        segments.append(signal[start:start + window_size_samples])
        start += stride_samples
    return segments


def load_raw_data_ptbxl(df, sampling_rate, path):
    if sampling_rate == 100:
        if os.path.exists(os.path.join(path, 'raw100.npy')):
            data = np.load(os.path.join(path, 'raw100.npy'), allow_pickle=True)
        else:
            data = [wfdb.rdsamp(os.path.join(path, 'raw', f)) for f in tqdm(df.filename_lr)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(os.path.join(path, 'raw100.npy'), 'wb'), protocol=4)
    elif sampling_rate == 500:
        if os.path.exists(os.path.join(path, 'raw500.npy')):
            data = np.load(os.path.join(path, 'raw500.npy'), allow_pickle=True)
        else:
            data = [wfdb.rdsamp(os.path.join(path, 'raw', f)) for f in tqdm(df.filename_hr)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(os.path.join(path, 'raw500.npy'), 'wb'), protocol=4)
    else:
        raise ValueError("不支持的采样率。请选择 100Hz 或 500Hz。")
    return data


def compute_label_aggregations(df, folder, ctype):
    df['scp_codes_len'] = df.scp_codes.apply(lambda x: len(x))

    aggregation_df = pd.read_csv(os.path.join(folder, 'raw', 'scp_statements.csv'), index_col=0)

    if ctype in ['diagnostic', 'subdiagnostic', 'superdiagnostic']:

        def aggregate_all_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    tmp.append(key)
            return list(set(tmp))

        def aggregate_subdiagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    c = diag_agg_df.loc[key].diagnostic_subclass
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        def aggregate_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    c = diag_agg_df.loc[key].diagnostic_class
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        diag_agg_df = aggregation_df[aggregation_df.diagnostic == 1.0]
        if ctype == 'diagnostic':
            df['diagnostic'] = df.scp_codes.apply(aggregate_all_diagnostic)
            df['diagnostic_len'] = df.diagnostic.apply(lambda x: len(x))
        elif ctype == 'subdiagnostic':
            df['subdiagnostic'] = df.scp_codes.apply(aggregate_subdiagnostic)
            df['subdiagnostic_len'] = df.subdiagnostic.apply(lambda x: len(x))
        elif ctype == 'superdiagnostic':
            df['superdiagnostic'] = df.scp_codes.apply(aggregate_diagnostic)
            df['superdiagnostic_len'] = df.superdiagnostic.apply(lambda x: len(x))
    elif ctype == 'form':
        form_agg_df = aggregation_df[aggregation_df.form == 1.0]

        def aggregate_form(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in form_agg_df.index:
                    c = key
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        df['form'] = df.scp_codes.apply(aggregate_form)
        df['form_len'] = df.form.apply(lambda x: len(x))
    elif ctype == 'rhythm':
        rhythm_agg_df = aggregation_df[aggregation_df.rhythm == 1.0]

        def aggregate_rhythm(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in rhythm_agg_df.index:
                    c = key
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        df['rhythm'] = df.scp_codes.apply(aggregate_rhythm)
        df['rhythm_len'] = df.rhythm.apply(lambda x: len(x))
    elif ctype == 'all':
        df['all_scp'] = df.scp_codes.apply(lambda x: list(set(x.keys())))

    return df


def select_data(XX, YY, ctype, min_samples):
    # convert multilabel to multi-hot
    mlb = MultiLabelBinarizer()

    if ctype == 'diagnostic':
        X = XX[YY.diagnostic_len > 0]
        Y = YY[YY.diagnostic_len > 0]
        mlb.fit(Y.diagnostic.values)
        y = mlb.transform(Y.diagnostic.values)
    elif ctype == 'subdiagnostic':
        counts = pd.Series(np.concatenate(YY.subdiagnostic.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.subdiagnostic = YY.subdiagnostic.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['subdiagnostic_len'] = YY.subdiagnostic.apply(lambda x: len(x))
        X = XX[YY.subdiagnostic_len > 0]
        Y = YY[YY.subdiagnostic_len > 0]
        mlb.fit(Y.subdiagnostic.values)
        y = mlb.transform(Y.subdiagnostic.values)
    elif ctype == 'superdiagnostic':
        counts = pd.Series(np.concatenate(YY.superdiagnostic.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.superdiagnostic = YY.superdiagnostic.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['superdiagnostic_len'] = YY.superdiagnostic.apply(lambda x: len(x))
        X = XX[YY.superdiagnostic_len > 0]
        Y = YY[YY.superdiagnostic_len > 0]
        mlb.fit(Y.superdiagnostic.values)
        y = mlb.transform(Y.superdiagnostic.values)
    elif ctype == 'form':
        # filter
        counts = pd.Series(np.concatenate(YY.form.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.form = YY.form.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['form_len'] = YY.form.apply(lambda x: len(x))
        # select
        X = XX[YY.form_len > 0]
        Y = YY[YY.form_len > 0]
        mlb.fit(Y.form.values)
        y = mlb.transform(Y.form.values)
    elif ctype == 'rhythm':
        # filter
        counts = pd.Series(np.concatenate(YY.rhythm.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.rhythm = YY.rhythm.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['rhythm_len'] = YY.rhythm.apply(lambda x: len(x))
        # select
        X = XX[YY.rhythm_len > 0]
        Y = YY[YY.rhythm_len > 0]
        mlb.fit(Y.rhythm.values)
        y = mlb.transform(Y.rhythm.values)
    elif ctype == 'all':
        # filter
        counts = pd.Series(np.concatenate(YY.all_scp.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.all_scp = YY.all_scp.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['all_scp_len'] = YY.all_scp.apply(lambda x: len(x))
        # select
        X = XX[YY.all_scp_len > 0]
        Y = YY[YY.all_scp_len > 0]
        mlb.fit(Y.all_scp.values)
        y = mlb.transform(Y.all_scp.values)
    else:
        pass

    return X, Y, y, mlb


def preprocess_signals(X_train, X_validation, X_test):
    # Standardize data such that mean 0 and variance 1
    ss = StandardScaler()
    ss.fit(np.vstack(X_train).flatten()[:, np.newaxis].astype(float))

    return apply_standardizer(X_train, ss), apply_standardizer(X_validation, ss), apply_standardizer(X_test, ss)


def apply_standardizer(X, ss):
    X_tmp = []
    for x in X:
        x_shape = x.shape
        X_tmp.append(ss.transform(x.flatten()[:, np.newaxis]).reshape(x_shape))
    X_tmp = np.array(X_tmp)
    return X_tmp


def data_slice(data):
    data_process = []
    for dat in data:
        if dat.shape[0] < 1000:
            dat = resample(dat, 1000, axis=0)
        elif dat.shape[0] > 1000:
            dat = dat[:1000, :]
        if dat.shape[1] != 12:
            dat = dat[:, 0:12]

        data_process.append(dat)
    return np.array(data_process)


# hf
def name2index(path):
    list_name = []
    for line in open(path, encoding='utf-8'):
        list_name.append(line.strip())
    name2indx = {name: i for i, name in enumerate(list_name)}
    return name2indx


def file2index(path, name2idx):
    file2index = dict()
    for line in open(path, encoding='utf-8'):
        arr = line.strip().split('\t')
        id = arr[0]
        labels = [name2idx[name] for name in arr[3:]]
        file2index[id] = labels
    return file2index


def load_raw_data_hf(root='../data/hf/', resample_num=1000, num_classes=34):
    if os.path.exists(os.path.join(root, 'raw100_data.npy')):
        data = np.load(os.path.join(root, 'raw100_data.npy'), allow_pickle=True)
        y = np.load(os.path.join(root, 'raw100_label.npy'), allow_pickle=True)
    else:
        name2idx = name2index(os.path.join(root, 'hf_round2_arrythmia.txt'))
        file2idx = file2index(os.path.join(root, 'hf_round2_label.txt'), name2idx)
        data, label = [], []
        for file, list_idx in file2idx.items():
            temp = np.zeros([5000, 12])
            df = pd.read_csv(os.path.join(root, 'hf_round2_train', file), sep=' ').values
            temp[:, 2] = df[:, 1] - df[:, 0]
            temp[:, 3] = -(df[:, 0] + df[:, 1]) / 2
            temp[:, 4] = df[:, 0] - df[:, 1] / 2
            temp[:, 5] = df[:, 1] - df[:, 0] / 2
            temp[:, 0:2] = df[:, 0:2]
            temp[:, 6:12] = df[:, 2:8]
            sig = resample(temp, resample_num)
            min_max_scaler = preprocessing.MinMaxScaler()
            ecg = min_max_scaler.fit_transform(sig)
            data.append(ecg)
            label.append(tuple(list_idx))
        data = np.array(data)
        pickle.dump(data, open(os.path.join(root, 'raw100_data.npy'), 'wb'), protocol=4)
        mlb = MultiLabelBinarizer(classes=[i for i in range(num_classes)])
        y = mlb.fit_transform(label)
        y = np.array(y)
        pickle.dump(y, open(os.path.join(root, 'raw100_label.npy'), 'wb'), protocol=4)
    return data, y


def hf_dataset(root='../data/hf/', resample_num=1000, num_classes=34):
    data, label = load_raw_data_hf(root, resample_num, num_classes)
    data_num = len(label)
    shuffle_ix = np.random.permutation(np.arange(data_num))
    data = data[shuffle_ix]
    labels = label[shuffle_ix]

    X_train = data[int(data_num * 0.2):int(data_num * 0.8)]
    y_train = labels[int(data_num * 0.2):int(data_num * 0.8)]

    X_val = data[int(data_num * 0.8):]
    y_val = labels[int(data_num * 0.8):]

    X_test = data[:int(data_num * 0.2)]
    y_test = labels[:int(data_num * 0.2)]

    return X_train, y_train, X_val, y_val, X_test, y_test
