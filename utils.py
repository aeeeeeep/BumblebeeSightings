import imghdr
import math
import os
from warnings import simplefilter

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm

simplefilter(action='ignore', category=FutureWarning)


def pad_or_cut(value: np.ndarray, target_length: int):
    """填充或截断一维numpy到固定的长度"""
    data_row = None
    if len(value) < target_length:  # 填充
        data_row = np.pad(value, [(0, target_length - len(value))])
    elif len(value) > target_length:  # 截断
        data_row = value[:target_length]
    return data_row


def data_split(full_list, ratio, shuffle=True):
    n_total = len(full_list)
    positive = full_list[full_list.label == 1]
    negative = full_list[full_list.label == 0]
    n_positive = int(len(positive) * ratio)
    n_negative = int(len(negative) * ratio)
    if n_total == 0 or n_positive < 1 or n_negative < 1:
        return [], full_list
    else:
        sublist_1_1 = positive.iloc[:n_positive, :]
        sublist_1_2 = negative.iloc[:n_negative, :]
        sublist_1 = pd.concat([sublist_1_1, sublist_1_2])
    sublist_2_1 = positive.iloc[n_positive:, :]
    sublist_2_2 = negative.iloc[n_negative:, :]
    sublist_2 = pd.concat([sublist_2_1, sublist_2_2])
    if shuffle:
        sublist_1 = sublist_1.sample(frac=1)
    sublist_1.reset_index(inplace=True)
    sublist_2.reset_index(inplace=True)
    return sublist_1, sublist_2


def check_error_img(path):
    original_images = []

    for root, dirs, filenames in os.walk(path):
        for filename in filenames:
            original_images.append(os.path.join(root, filename))

    original_images = sorted(original_images)
    print('num:', len(original_images))
    error_images = []
    for filename in tqdm(original_images):
        check = imghdr.what(filename)
        if check == None:
            error_images.append(filename)
    if len(error_images) == 0:
        print('All images are normal!')
    else:
        for i in error_images:
            print(i)
        print('{} error images'.format(len(error_images)))
        exit()


def process_notes(notes):
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = stopwords.words('english')
    notes_processed = [[word.lower() for word in tokenizer.tokenize(x) if word not in stop_words] for x in notes]
    return notes_processed

def cal_weight(x):
    '''熵值法计算变量的权重'''
    # 标准化
    # x = x.apply(lambda x: ((x - np.min(x)) / (np.max(x) - np.min(x))))

    # 求k
    rows = x.index.size  # 行
    cols = x.columns.size  # 列
    k = 1.0 / math.log(rows)

    lnf = [[None] * cols for i in range(rows)]

    # 矩阵计算--
    # 信息熵
    # p=array(p)
    x = np.array(x)
    lnf = [[None] * cols for i in range(rows)]
    lnf = np.array(lnf)
    for i in range(0, rows):
        for j in range(0, cols):
            if x[i][j] == 0:
                lnfij = 0.0
            else:
                p = x[i][j] / x.sum(axis=0)[j]
                lnfij = math.log(p) * p * (-k)
            lnf[i][j] = lnfij
    lnf = pd.DataFrame(lnf)
    E = lnf

    # 计算冗余度
    d = 1 - E.sum(axis=0)
    # 计算各指标的权重
    w = [[None] * 1 for i in range(cols)]
    for j in range(0, cols):
        wj = d[j] / sum(d)
        w[j] = wj
        # 计算各样本的综合得分,用最原始的数据

    w = pd.DataFrame(w)
    return w
