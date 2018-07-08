import pandas as pd
import numpy as np
from pandas import Series

def most_similar(hero_name,embeddings,hero_dict):
    '''
    用于英雄相似度计算，对给定英雄，计算出与其最相近10个英雄并给出相似度
    :param hero_name: 英雄名字
    :param embeddings: 训练好的模型参数
    :param hero_dict: 英雄字典
    :return: 最相近的英雄及其相似度
    '''
    #正则化参数
    normalized_embeddings = embeddings / (embeddings ** 2).sum(axis=1).reshape((-1, 1)) ** 0.5

    #将英雄字典的key和value互换，新字典可以通过英雄名来访问英雄id
    new_dict = {v:k for k,v in hero_dict.items()}
    #allname列表用于判断输入的英雄是否在字典内
    allname = []
    for id,name in hero_dict.items():
        allname.append(name)
    if hero_name in allname:
        w = new_dict[hero_name]
        v = normalized_embeddings[w]
        sims = np.dot(normalized_embeddings, v)
        sort = sims.argsort()[::-1]
        sort = sort[sort > 0]
        return pd.Series([(hero_dict[i],sims[i]) for i in sort[:10]])
    else:
        return '请输入正确的英雄名称'