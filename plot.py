import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

def plot_embeddings(embeddings,hero_dict):
    '''
    将训练好的嵌入层，用pca降维。
    并用matplotlib进行可视化展示
    :param embeddings:训练好的英雄向量
    :param hero_dict:英雄名和id字典
    '''
    #从字典中获取所有英雄的名称和id
    allid, allname = [], []
    for id, name in hero_dict.items():
        allid.append(id)
        allname.append(name)

    embeddings -= np.mean(embeddings, axis=0)

    # 用来正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']

    # 用pca将数据降至2维
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    x, y = embeddings_2d[:, 0], embeddings_2d[:, 1]
    #只需画出已有id的英雄数据
    allid_x, allid_y = x[allid], y[allid]
    fig = plt.figure(figsize=(20, 16), dpi=100)
    ax = plt.subplot(111)
    marker_size = 10
    ax.scatter(allid_x, allid_y, c='tomato', s=marker_size)
    print('done')
    for i in range(len(allid_x)):
        #将英雄名称标注在图上数据点的位置
        ax.annotate(allname[i], (allid_x[i], allid_y[i]), fontsize=10)
    plt.show()
    fig.savefig('./dota2hero_embddings_2d.png')