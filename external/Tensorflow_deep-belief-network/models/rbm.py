# -*- coding: utf-8 -*-
import tensorflow as tf
import sys
sys.path.append("../base")
from base_func import Batch

class RBM(object):
    def __init__(self,
                 rbm_v_type='bin',
                 rbm_struct=[784,100],
                 rbm_epochs=10,
                 batch_size=32,
                 cd_k=1,
                 rbm_lr=1e-3):
        self.rbm_v_type=rbm_v_type
        self.n_v = rbm_struct[0]
        self.n_h = rbm_struct[1]
        self.rbm_epochs = rbm_epochs
        self.batch_size = batch_size
        self.cd_k = cd_k
        self.rbm_lr = rbm_lr
        
    ###################
    #    RBM_model    #
    ###################
    
    def build_model(self):
        # feed 变量
        self.input_data = tf.placeholder(tf.float32, [None, self.n_v]) # N等于batch_size（训练）或_num_examples（测试）
        # 权值 变量（初始化）
        self.W = tf.Variable(tf.truncated_normal(shape=[self.n_v, self.n_h], stddev=0.1), name='W')
        self.bh = tf.Variable(tf.constant(0.1, shape=[self.n_h]),name='bh')
        self.bv = tf.Variable(tf.constant(0.1, shape=[self.n_v]),name='bv')
        self.parameter = [self.W, self.bh]
        # v0,h0
        v0=self.input_data # v0
        h0,s_h0=self.transform(v0) # h0
        # vk,hk
        vk=self.reconstruction(s_h0) # v1
        for k in range(self.cd_k-1):
            _,s_hk=self.transform(vk) # trans（sample）
            vk=self.reconstruction(s_hk) # recon（compute）
        hk,_=self.transform(vk) # hk
        # upd8
        positive=tf.matmul(tf.expand_dims(v0,-1), tf.expand_dims(h0,1))
        negative=tf.matmul(tf.expand_dims(vk,-1), tf.expand_dims(hk,1))
        self.w_upd8 = self.W.assign_add(tf.multiply(self.rbm_lr, tf.reduce_mean(tf.subtract(positive, negative), 0)))
        self.bh_upd8 = self.bh.assign_add(tf.multiply(self.rbm_lr, tf.reduce_mean(tf.subtract(h0, hk), 0)))
        self.bv_upd8 = self.bv.assign_add(tf.multiply(self.rbm_lr, tf.reduce_mean(tf.subtract(v0, vk), 0)))
        self.train_batch_cdk = [self.w_upd8, self.bh_upd8, self.bv_upd8]
        # loss
        self.loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(v0, vk))))
            
    def train_model(self,train_X,sess):
        # 初始化变量
        sess.run(tf.global_variables_initializer())
        _data=Batch(images=train_X,
                    batch_size=self.batch_size)
        n=train_X.shape[0]
        # 迭代次数
        for i in range(self.rbm_epochs):
            # 批次训练
            for _ in range(int(n/self.batch_size)): 
                batch = _data.next_batch()
                loss,_= sess.run([self.loss,self.train_batch_cdk],feed_dict={self.input_data: batch})
            print('>>> epoch = {} , loss = {:.4}'.format(i+1,loss))
    
    def transform(self,v):
        z = tf.add(tf.matmul(v, self.W), self.bh)
        prob_h=tf.nn.sigmoid(z) # compute
        state_h= tf.to_float(tf.random_uniform([tf.shape(v)[0],self.n_h])<prob_h) # sample
#        prob_h=z
#        state_h=tf.add(prob_h, tf.random_uniform([tf.shape(v)[0],self.n_h]))
        return prob_h,state_h
    
    def reconstruction(self,h):
        z = tf.add(tf.matmul(h, tf.transpose(self.W)), self.bv)
        if self.rbm_v_type=='bin':
            prob_v=tf.nn.sigmoid(z)
        else:
            prob_v=z
        return prob_v