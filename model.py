from keras.models import Model
from keras.layers import Input, Dense, Activation,Embedding,Lambda
import keras as K

def build_model(emb_dim=20,dense_dim=248):
    '''
    建立一个由嵌入层，一个全连接层，以及一个softmax层组成的模型。
    模型输入为同队伍的4个英雄，用于预测最后一个位置出现的英雄。
    :param
        emb_dim:用于控制嵌入层的数据维数，数据格式int
        dense_dim:用于控制全连接层的units数量
    :return:keras的函数式模型
    '''
    input_data = Input(shape=(4, ))
    #嵌入层
    emb = Embedding(input_dim=125, output_dim=emb_dim, input_length=4)(input_data)
    #将嵌入层的数据进行求和
    emb_sum = Lambda(lambda x: K.backend.sum(x, axis=1))(emb)
    #全连接层
    dense1 = Dense(dense_dim, activation="relu")(emb_sum)
    #softmax层
    dense2 = Dense(125, activation="softmax")(dense1)
    model = Model(inputs=input_data, outputs=dense2)
    return model