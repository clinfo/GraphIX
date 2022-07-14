import tensorflow as tf
import numpy as np
import joblib
import layers
from modules.default_model import DefaultModel
import tensorflow.contrib.keras as K
from tensorflow.contrib.keras import backend

class KGNetwork(DefaultModel):
    def __init__(self, feed_embedded_layer):
        self.feed_embedded_layer = feed_embedded_layer

    @classmethod
    def build_placeholders(cls, info, config, batch_size=4, feed_embedded_layer=False):
        adj_channel_num=info.adj_channel_num
        placeholders = {
            'adjs':[[tf.sparse_placeholder(tf.float32,name="adj_"+str(a)+"_"+str(b)) for a in range(adj_channel_num)] for b in range(batch_size)],
            'nodes': tf.placeholder(tf.int32, shape=(batch_size,info.graph_node_num),name="node"),
            'mask': tf.placeholder(tf.float32, shape=(batch_size,),name="mask"),
            'dropout_rate': tf.placeholder(tf.float32, name="dropout_rate"),
            'enabled_node_nums': tf.placeholder(tf.int32, shape=(batch_size,), name="enabled_node_nums"),
            'is_train': tf.placeholder(tf.bool, name="is_train"),
            'embedded_layer': tf.placeholder(tf.float32, shape=(batch_size, info.all_node_num, config["embedding_dim"]), 
                                             name="embedded_layer") # modified
        }

        placeholders['preference_label_list']= tf.placeholder(tf.int64, shape=(batch_size,None,6),name="preference_label_list")
        if info.feature_enabled:
            placeholders['features']=tf.placeholder(tf.float32, shape=(batch_size,info.graph_node_num,info.feature_dim),name="feature")
        else:
            placeholders['features']=None
        placeholders['embedded_layer'] = tf.placeholder(tf.float32, shape=(batch_size, info.all_node_num,
                                                                           config["embedding_dim"],), name="embedded_layer")
        cls.placeholders = placeholders
        cls.all_node_num = info.all_node_num
        cls.embedding_dim = config["embedding_dim"]
        return  placeholders

    def build_model(self, placeholders, info, config, batch_size=4):
        adj_channel_num=info.adj_channel_num
        embedding_dim=config["embedding_dim"]
        in_adjs=placeholders["adjs"]
        features=placeholders["features"]
        in_nodes=placeholders["nodes"]
        label_list=placeholders["preference_label_list"]
        mask=placeholders["mask"]
        enabled_node_nums=placeholders["enabled_node_nums"]
        is_train=placeholders["is_train"]
        embedded_layer = placeholders['embedded_layer']
        layer=features
        input_dim=info.feature_dim

        self.embedding_layer = K.layers.Embedding(info.all_node_num,embedding_dim)(in_nodes) # modified
        if features is None:
            if self.feed_embedded_layer:
                layer = embedded_layer
            else:
                layer = self.embedding_layer # modified
        else: # modified
            layer = features # modified
        # layer: batch_size x graph_node_num x dim
        layer=layers.GraphConv(64,adj_channel_num)(layer,adj=in_adjs)
        layer=tf.nn.tanh(layer)
        #
        prediction=layer
        lp_prediction=tf.matmul(layer,layer,transpose_b=True)
        pred0=tf.gather(prediction[0,:,:],label_list[0,:,0])
        pred1=tf.gather(prediction[0,:,:],label_list[0,:,2])
        pred2=tf.gather(prediction[0,:,:],label_list[0,:,3])
        pred3=tf.gather(prediction[0,:,:],label_list[0,:,5])
        s1=tf.reduce_sum(pred0*pred1,axis=1)
        s2=tf.reduce_sum(pred2*pred3,axis=1)
        #score = s2 - s1
        score = s1 - s2
        self.score = s1
        #output=1.0/(1.0+tf.exp(score))
        output = tf.nn.sigmoid(score)
        cost=-1*tf.log(output+1.0e-10)
        self.loss = cost
        # computing cost and metrics
        #cost=mask*tf.reduce_mean(cost,axis=1)
        print(cost)
        cost_opt=tf.reduce_mean(cost)
        metrics={}
        cost_sum=tf.reduce_sum(cost)
        ####
        pre_count=tf.cast(tf.greater(s1,s2),tf.float32)
        metrics["correct_count"]=tf.reduce_sum(pre_count)
        ###
        count=tf.shape(label_list[0,:,0])[0]
        metrics["count"]=count
        self.out=prediction
        return self, lp_prediction, cost_opt, cost_sum, metrics

    def embedding(self, sess, data=None): # modified
        key = self.placeholders['nodes'] # modified
        return sess.run(self.embedding_layer, feed_dict = {key: data}) # modified

build_placeholders = KGNetwork.build_placeholders

def build_model(placeholders, info, config, batch_size=4, feed_embedded_layer=False):
    net = KGNetwork(feed_embedded_layer)
    return net.build_model(placeholders, info, config, batch_size=batch_size)
