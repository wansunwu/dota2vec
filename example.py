from dataset import get_hero_dict
from model import build_model
from plot import plot_embeddings
from evaluation import most_similar

#用已经训练好的模型参数来生成dota2vec的可视化图表已经测试相似度

#读取训练好的参数
model = build_model()
model.load_weights("dota2vec.h5")
embeddings = model.get_weights()[0]

#读取英雄名字典
hero_dict = get_hero_dict()

#绘制嵌入层的可视化图
plot_embeddings(embeddings,hero_dict)

#测试英雄相似度
hero_name = '水晶室女'
print(most_similar(hero_name,embeddings,hero_dict))