import tensorflow as tf
import os
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets

# 情感分类

batchsz = 128  # 批量大小
total_words = 10000  # 词汇表大小 N_vocab
max_review_len = 80  # 句子最大长度 s，大于的句子部分将截断，小于的将填充
embedding_len = 100  # 词向量特征长度 n
# 加载 IMDB 数据集，此处的数据采用数字编码，一个数字代表一个单词
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=total_words)
'''
# 数字编码表
word_index = keras.datasets.imdb.get_word_index()
# 打印出编码表的单词和对应的数字
for k,v in word_index.items():
print(k,v)
# 翻转编码表
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
对于一个数字编码的句子，通过如下函数转换为字符串数据：
def decode_review(text):
 return ' '.join([reverse_word_index.get(i, '?') for i in text])
例如转换某个句子，代码如下：
decode_review(x_train[0])
'''

# 截断和填充句子，使得等长，此处长句子保留句子后面的部分，短句子在前面填充
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_review_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_review_len)

# 构建数据集，打散，批量，并丢掉最后一个不够 batchsz 的 batch
db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
db_train = db_train.shuffle(1000).batch(batchsz, drop_remainder=True)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.batch(batchsz, drop_remainder=True)

'''
----------------网络---------------
'''


class MyRNN(keras.Model):
    def __init__(self, units):
        super(MyRNN, self).__init__()
        # [b, 64]，构建 Cell 初始化状态向量，重复使用
        self.state0 = [tf.zeros([batchsz, units])]
        self.state1 = [tf.zeros([batchsz, units])]
        self.embedding = layers.Embedding(total_words, embedding_len, input_length=max_review_len)

        self.cell1 = layers.SimpleRNNCell(units, dropout=0.2)
        self.cell2 = layers.SimpleRNNCell(units, dropout=0.2)

        self.out = layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        x = inputs
        x = self.embedding(x)

        state0 = self.state0
        state1 = self.state1

        for word in tf.unstack(x, axis=1):
            out0, state0 = self.cell1(word, state0, training)
            out1, state1 = self.cell2(out0, state1, training)
        out = self.out(out1)
        out = tf.sigmoid(out)
        return out


# train
def train():
    units = 64  # 状态向量长度
    epoch = 2

    optim = optimizers.Adam()
    model = MyRNN(units)

    model.compile(optimizer=optim, loss=keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
    his = model.fit(db_train, epochs=epoch, validation_data=db_test)
    # 一个 History 对象。其 History.history 属性是连续 epoch 训练损失和评估值，以及验证集损失和评估值的记录（如果适用）
    print(his.history)
    # 测试
    result = model.evaluate(db_test)
    print(result)
    # 保存
    model.save('the_save_model', save_format="tf")


# test
def test():
    model = MyRNN(64)
    model.load_weights('a.ckpt')
    model = keras.models.load_model('the_save_model')
    for i, (x, y) in enumerate(db_test):
        # 得到句子
        import json
        with open('imdb_word_index.json') as f:
            word_index = json.load(f)
        # 翻转编码表
        reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
        word = []
        # 输出其中一个测试句子
        for j in x[2]:
            word.append(reverse_word_index.get(int(j)))
        print("测试句子2", word)

        y_pre = model(x)
        # print("预测",y_pre)
        print("真实", y)

        right = 0
        for k in range(128):
            if y_pre[k] >= 0.5 and y[k] == 1:
                right += 1
            if y_pre[k] < 0.5 and y[k] == 0:
                right += 1
        print(right)
        break


if __name__ == '__main__':
    test()
