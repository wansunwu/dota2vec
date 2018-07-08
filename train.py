import pandas as pd
import numpy as np
from pandas import Series,DataFrame
import tensorflow as tf
import keras as K
from dataset import dataset_to_features,get_hero_dict
from model import build_model

#读取数据
dota_data = pd.read_csv('vh_game_data.csv',index_col=0)
train_data = dataset_to_features(dota_data)

x,y = train_data
#将y转换成125维的one-hot编码
y = K.utils.to_categorical(y, 125)

#获取模型
model = build_model()
opt = K.optimizers.RMSprop(lr=0.0001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(
    optimizer=opt,
    loss='mse',
    metrics=['accuracy'])

#训练模型
model.fit(x,y,batch_size=200,epochs=50,shuffle=True,verbose=1,validation_split=0.3)

#读取并保存模型参数
embeddings = model.get_weights()[0]
model.save_weights("dota2vec.h5")


