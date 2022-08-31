import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import argparse
import psutil
import tensorlayer as tl
import tensorflow as tf
import numpy as np

from model import Vgg19_simple_api
from flower_data import Dataset



class Main():
    '''
    项目中必须继承FlyAI类，否则线上运行会报错。
    '''
    def __init__(self):
        self.bottleneck_size = 1000
        self.num_class = 5

    def memory_usage(self):
        mem_available = psutil.virtual_memory().available
        mem_process = psutil.Process(os.getpid()).memory_info().rss
        return round(mem_process / 1024 / 1024, 2), round(mem_available / 1024 / 1024, 2)
    def train(self):
        batch_size=16
        max_steps=5000
        data=Dataset()

        input_image = tf.placeholder(dtype=tf.float32,shape=[None, 224, 224, 3],name='input_image')
        label_image=tf.placeholder(dtype=tf.float32, shape=[None,5], name='input_image')
        net_vgg, vgg_target_emb = Vgg19_simple_api(input_image, reuse=False)
        ###============================= Create Model ===============================###
        weights = tf.Variable(tf.truncated_normal([self.bottleneck_size, self.num_class], stddev=0.1), name="fc9/w")
        biases = tf.Variable(tf.constant(0.1, shape=[self.num_class]), name="fc9/b")
        logits = tf.matmul(net_vgg.outputs, weights) + biases
        # 计算损失、准确率
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(label_image, 1), logits=logits))
        # loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_image,logits=net_vgg.outputs))
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(label_image, 1), tf.argmax(logits, 1)), tf.float32))
        # 优化器
        optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.00001).minimize(loss=loss)
        # optimizer=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss=loss)
        ###============================= LOAD VGG ===============================###
        vgg19_npy_path = "vgg19.npy"
        if not os.path.isfile(vgg19_npy_path):
            print("Please download vgg19.npz from : https://github.com/machrisaa/tensorflow-vgg")
            exit()
        npz = np.load(vgg19_npy_path, encoding='latin1').item()

        params = []
        for val in sorted(npz.items()):
            W = np.asarray(val[1][0])
            b = np.asarray(val[1][1])
            print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
            params.extend([W, b])
        print('ok')
        saver = tf.train.Saver(var_list=[weights, biases])
        init_op = tf.global_variables_initializer()
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        with tf.Session(config=config) as sess:
            tl.layers.initialize_global_variables(sess)
            sess.run(init_op)
            tl.files.assign_params(sess, params, net_vgg)
            ###========================== RESTORE MODEL =============================###
            # tl.files.load_and_assign_npz(sess=sess,name="data/output/model/my_vgg.npz",network=net_vgg)
            tl.files.load_and_assign_npz(sess=sess, name=os.path.join('model','my_vgg.npz'), network=net_vgg)
            try:
                saver.restore(sess=sess,save_path=os.path.join('model','model.ckpt'))
            except:
                print("load model err")
            ###========================== TRAIN MODEL =============================###
            val_image,val_label=data.validation()
            validate_feed = {input_image:val_image, label_image:val_label}
            for epoch in range(max_steps):
                # 模型训练
                train_image,train_label=data.train_next_batch(batch_size)
                train_feed = {input_image:train_image, label_image:train_label}
                cur_loss,_=sess.run([loss, optimizer],feed_dict=train_feed)   #返回的变量名与sess.run()中的不可以相同
                print("Epoch:%d,loss %.2f"%(epoch,cur_loss*100),self.memory_usage())
                if epoch!=0 and epoch%50==0:
                    # 模型验证
                    my_accuracy=sess.run(accuracy,feed_dict=validate_feed)
                    # print(my_accuracy)
                    print("Epoch:%d ,accuracy: %.2f%%"%(epoch, my_accuracy*100))
                    #模型保存
                    tl.files.save_npz(net_vgg.all_params, name=os.path.join('model','my_vgg.npz'),sess=sess)
                    saver.save(sess=sess, save_path=os.path.join('model', 'model.ckpt'))
    def test(self):
        data = Dataset()
        input_image = tf.placeholder('float32', [None, 224, 224, 3], name='image_input')
        label_image = tf.placeholder('float32', [None, 5], name='input_image')
        # net_vgg, vgg_target_emb = Vgg19_simple_api((input_image + 1) / 2, reuse=False)
        net_vgg, vgg_target_emb = Vgg19_simple_api(input_image, reuse=False)
        ###============================= Create Model ===============================###
        weights = tf.Variable(tf.truncated_normal([self.bottleneck_size, self.num_class], stddev=0.1), name="fc9/w")
        biases = tf.Variable(tf.constant(0.1, shape=[self.num_class]), name="fc9/b")
        logits = tf.matmul(net_vgg.outputs, weights) + biases
        # model_out=my_model(net_vgg.outputs,reuse=False)
        # 计算损失、准确率
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(label_image, 1), logits=logits))
        # loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_image,logits=net_vgg.outputs))
        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(label_image, 1), tf.argmax(logits, 1)), tf.float32))
        # print(accuracy)
        # 优化器
        # optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.00001).minimize(loss=loss)
        # optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss=loss)
        ###============================= LOAD VGG ===============================###

        saver = tf.train.Saver(var_list=[weights, biases])
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        with tf.Session(config=config) as sess:
            ###========================== RESTORE MODEL =============================###
            tl.files.load_and_assign_npz(sess=sess, name=os.path.join('model', 'my_vgg.npz'), network=net_vgg)
            # tl.files.load_and_assign_npz(sess=sess, name=os.path.join(MODEL_PATH, 'my_vgg_fc9.npz'), network=model_out)
            # if os.path.exists(os.path.join(MODEL_PATH, 'model.ckpt')):
            saver.restore(sess=sess, save_path=os.path.join('model', 'model.ckpt'))
            ###========================== TEST MODEL =============================###
            test_image, test_label = data.test()
            my_accuracy = 0
            for x_test, y_test in zip(test_image, test_label):
                validate_feed = {input_image: [x_test], label_image: [y_test]}
                # 模型验证
                acc = sess.run(accuracy, feed_dict=validate_feed)  # 返回的变量名与sess.run()中的不可以相同
                my_accuracy += acc
                print(my_accuracy)
            print("Accuracy %.2f" % (my_accuracy / len(test_image)))

if __name__ == '__main__':

    main = Main()
    main.train()
    main.test()