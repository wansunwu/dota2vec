import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import json

def dataset_to_features(dataset_df):
    '''
    将Dataframe中的比赛数据转化为训练所需的数据格式。
    每场比赛中天辉夜魇各5名英雄，共可生成10条用于训练的数据
    :param dataset_df:原比赛数据，格式为Dataframe
    :return:用于训练的目标矩阵
    '''
    # 构造一个空的x目标矩阵，列数为4，行数为样本数量*10
    x_matrix = np.zeros((dataset_df.shape[0] * 5 * 2, 4))

    # 构造一个空的y目标矩阵，行数为样本数量*10
    y_matrix = np.zeros(dataset_df.shape[0] * 5 * 2)

    # 将原样本中的数据，用pandas的values函数导出为一个numpy的矩阵类型
    dataset_np = dataset_df.values

    # 对矩阵的每行每个英雄，分别映射到目标矩阵中
    for i, row in enumerate(dataset_np):
        radiant_heroes = row[1:6]
        dire_heroes = row[6:11]
        for j in range(5):
            y_matrix[i * 10 + j] = radiant_heroes[j]
            y_matrix[i * 10 + j + 5] = dire_heroes[j]
            x_matrix[i * 10 + j] = np.delete(radiant_heroes, j, axis=0)
            x_matrix[i * 10 + j + 5] = np.delete(dire_heroes, j, axis=0)
    return [x_matrix, y_matrix]

def get_hero_dict(en_name=False):
    '''
    从英雄的基础数据中获取英雄id和名字的对应字典
    :param en_name:是否获取英文名称
    :return:hero_dict:
    '''
    with open('hero_basic.json', 'r', encoding='UTF-8') as json_file:
        json_data = json.load(json_file)
    hero_dict = dict()

    if en_name:
        for entry in json_data['data']:
            hero_dict[entry['hero_id']] = entry['en_name']
    else:
        for entry in json_data['data']:
            hero_dict[entry['hero_id']] = entry['cn_name']
    return hero_dict