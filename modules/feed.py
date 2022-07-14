import tensorflow as tf
import numpy as np
import joblib
import time
import json

def get_all_label(config,info,label_list):
    num_label_list=len(label_list[0])
    dim=len(label_list[0][0])
    all_l=set()
    for b in range(len(label_list)):
        l1=set(label_list[0,:,0])
        l2=set(label_list[0,:,2])
        all_l=l1 | l2 | all_l
    return list(all_l)

def get_label_list_feed(config,info,label_list,label_itr,batch_size):
    num_label_list=len(label_list[0])
    dim=len(label_list[0][0])
    if config is not None and "label_batch_size" in config and label_itr is not None:
        num=config["label_batch_size"]
        j=label_itr
        temp_labels=np.zeros((batch_size,num,dim),dtype=np.int32)
        for b in range(len(label_list)):
            temp_labels[b,:num,:]=label_list[b][num*j:num*(j+1),:]
    else:
        temp_labels=np.zeros((batch_size,num_label_list,dim),dtype=np.int32)
        for b in range(len(label_list)):
            temp_labels[b,:len(label_list[b]),:]=label_list[b][:,:]
    return temp_labels

def get_preference_label_list_feed(config,info,label_list,label_itr,batch_size,nodes):
    all_label=get_all_label(config,info,label_list)
    num_label_list=len(label_list[0])
    dim=len(label_list[0][0])
    if config is not None and "label_batch_size" in config and label_itr is not None:
        num=config["label_batch_size"]
        j=label_itr
        temp_labels=np.zeros((batch_size,num,dim),dtype=np.int32)
        for b in range(len(label_list)):
            temp_labels[b,:num,:]=label_list[b][num*j:num*(j+1),:]
            if "preference_pair_mode" in config:
                if config["preference_pair_mode"]=="right":
                    neg=np.random.choice(all_label, (num,))
                    temp_labels[b,:num,5]=neg
                    temp_labels[b,:num,3]=temp_labels[b,:num,0]
                elif config["preference_pair_mode"]=="left":
                    neg=np.random.choice(all_label, (num,))
                    temp_labels[b,:num,3]=neg
                    temp_labels[b,:num,5]=temp_labels[b,:num,2]
                else:#both
                    temp_labels[b,:num,3]=temp_labels[b,:num,0]
                    temp_labels[b,:num,5]=temp_labels[b,:num,2]
                    neg=np.random.choice(all_label, (num,))
                    idx=np.random.choice([3,5], (num,))
                    temp_labels[b,:num,idx]=neg
            else:
                neg=np.random.choice(all_label, (num,))
                temp_labels[b,:num,5]=neg
                temp_labels[b,:num,3]=temp_labels[b,:num,0]
    else:
        temp_labels=np.zeros((batch_size,num_label_list,dim),dtype=np.int32)
        for b in range(len(label_list)):
            temp_labels[b,:,:]=label_list[b][:,:]
            num=len(label_list[b])
            if "preference_pair_mode" in config:
                if config["preference_pair_mode"]=="right":
                    neg=np.random.choice(all_label, (num,))
                    temp_labels[b,:,5]=neg
                    temp_labels[b,:,3]=temp_labels[b,:,0]
                elif config["preference_pair_mode"]=="left":
                    neg=np.random.choice(all_label, (num,))
                    temp_labels[b,:,3]=neg
                    temp_labels[b,:,5]=temp_labels[b,:,2]
                else:#both
                    temp_labels[b,:,3]=temp_labels[b,:,0]
                    temp_labels[b,:,5]=temp_labels[b,:,2]
                    neg=np.random.choice(all_label, (num,))
                    idx=np.random.choice([3,5], (num,))
                    temp_labels[b,:,idx]=neg
            else:
                if(len(nodes.shape)==2):
                    neg=np.random.choice(all_label, (num,))
                    temp_labels[b,:,5]=neg
                    temp_labels[b,:,3]=temp_labels[b,:,0]
                elif(len(nodes.shape)==3):
                    temp_labels[b,:,3]=np.random.choice(nodes[b,nodes[b,:,1]==0,0], (num,))
                    temp_labels[b,:,5]=np.random.choice(nodes[b,nodes[b,:,1]==1,0], (num,))
    return temp_labels



def construct_feed(batch_idx,placeholders,data,batch_size=None,dropout_rate=0.0,is_train=False, info=None, scaling=1.0, config=None, label_itr=None,ig_targets=[],**kwargs):
    adjs=data.adjs
    features=data.features
    nodes=data.nodes
    labels=data.labels
    mask_label=data.mask_label
    node_label=data.node_label
    mask_node_label=data.mask_node_label
    label_list=data.label_list
    sequences=data.sequences
    sequences_len=data.sequences_len
    vector_modal=data.vector_modal
    enabled_node_nums=data.enabled_node_nums

    feed_dict={}
    if batch_size is None:
        batch_size=len(batch_idx)
    for key,pl in placeholders.items():
        if key=="adjs":
            b_shape=None
            for b,b_pl in enumerate(pl):
                for ch,ab_pl in enumerate(b_pl):
                    if b <len(batch_idx):
                        bb=batch_idx[b]
                        b_shape=adjs[bb][ch][2]
                        if 'adjs' in ig_targets:
                            feed_dict[ab_pl]=tf.SparseTensorValue(adjs[bb][ch][0],adjs[bb][ch][1]*scaling,adjs[bb][ch][2])#*scaling
                        else:
                            feed_dict[ab_pl]=tf.SparseTensorValue(adjs[bb][ch][0],adjs[bb][ch][1],adjs[bb][ch][2])
                    else:
                        dummy_idx=np.zeros((0,2),dtype=np.int32)
                        dummy_val=np.zeros((0,),dtype=np.float32)
                        feed_dict[ab_pl]=tf.SparseTensorValue(dummy_idx,dummy_val,b_shape)
        elif key=="features" and features is not None:
            temp_features=np.zeros((batch_size,features.shape[1],features.shape[2]),dtype=np.float32)
            temp_features[:len(batch_idx),:,:]=features[batch_idx,:,:]
            if key in ig_targets:
                feed_dict[pl]=temp_features*scaling #features[batch_idx,:,:]
            else:
                feed_dict[pl]=temp_features
        elif key=="nodes" and features is None:
            temp_nodes=np.zeros((batch_size,nodes.shape[1]),dtype=np.int32)
            if(len(nodes.shape)==2):
                temp_nodes[:len(batch_idx),:]=nodes[batch_idx,:]
            elif(len(nodes.shape)==3):
                temp_nodes[:len(batch_idx),:]=nodes[batch_idx,:,0]
            feed_dict[pl]=temp_nodes #nodes[batch_idx,:]
        elif key=="labels":
            if len(labels.shape)==1:
                labels=labels[:,np.newaxis]
            if config is not None and "task" in config and config["task"]=="regression":
                temp_labels=np.zeros((batch_size,labels.shape[1]),dtype=np.float32)
            else:
                temp_labels=np.zeros((batch_size,labels.shape[1]),dtype=np.int32)

            temp_labels[:len(batch_idx),:]=labels[batch_idx,:]
            feed_dict[pl]=temp_labels #labels[batch_idx,:]
        elif key=="mask":
            mask=np.zeros((batch_size,),np.float32)
            mask[:len(batch_idx)]=1
            feed_dict[pl]=mask
        elif key=="mask_label":
            if len(mask_label.shape)==1:
                mask_label=mask_label[:,np.newaxis]
            temp_mask_label=np.zeros((batch_size,labels.shape[1]),np.float32)
            temp_mask_label[:len(batch_idx),:]=mask_label[batch_idx,:]
            feed_dict[pl]=temp_mask_label
        elif key=="node_label":
            temp_labels=np.zeros((batch_size,node_label.shape[1],node_label.shape[2]),dtype=np.float32)
            temp_labels[:len(batch_idx),:]=node_label[batch_idx,:,:]
            feed_dict[pl]=temp_labels #labels[batch_idx,:]
        elif key=="mask_node_label":
            #sgd_mask=np.random.random_sample((batch_size,mask_node_label.shape[1]))
            temp_labels=np.zeros((batch_size,mask_node_label.shape[1]),dtype=np.float32)
            temp_labels[:len(batch_idx),:]=mask_node_label[batch_idx,:]
            #sgd_mask[sgd_mask>=0.8]=1
            #sgd_mask[sgd_mask<0.8]=0
            feed_dict[pl]=temp_labels#*sgd_mask #labels[batch_idx,:]
        elif key=="label_list":
            feed_dict[pl]=get_label_list_feed(config,info,label_list,label_itr,batch_size)
        elif key=="preference_label_list":
            ll=get_preference_label_list_feed(config,info,label_list,label_itr,batch_size,nodes)
            feed_dict[pl]=ll
        elif key=="dropout_rate":
            feed_dict[pl]=dropout_rate
        elif key=="is_train":
            feed_dict[pl]=is_train
        elif key=="sequences" and sequences is not None:
            seqs=np.zeros((batch_size,sequences.shape[1]),np.int32)
            seqs[:len(batch_idx),:]=sequences[batch_idx,:]
            feed_dict[pl]=seqs
        elif key=="sequences_len" and sequences_len is not None:
            seqs_len=np.zeros((batch_size,2),np.int32)
            seqs_len[:len(batch_idx),1]=sequences_len[batch_idx]-1
            seqs_len[:len(batch_idx),0]=range(len(batch_idx))
            feed_dict[pl]=seqs_len
        elif info is not None and key in info.vector_modal_name and vector_modal is not None:
            j=info.vector_modal_name[key]
            vecs=np.zeros((batch_size,vector_modal[j].shape[1]),np.float32)
            vecs[:len(batch_idx),:]=vector_modal[j][batch_idx,:]
            if key in ig_targets:
                feed_dict[pl]=vecs*scaling
            else:
                feed_dict[pl]=vecs
        elif key=="mask_node" and enabled_node_nums is not None:
            temp_mask_label=np.zeros((batch_size,info.graph_node_num),np.float32)
            lengths=enabled_node_nums[batch_idx]
            for j,l in enumerate(lengths):
                temp_mask_label[j,:l]=1.0
            feed_dict[pl]=temp_mask_label
        elif key=="enabled_node_nums" and enabled_node_nums is not None:
            temp_enabled_node_nums=np.zeros((batch_size, ), np.int32)
            temp_enabled_node_nums[:len(batch_idx)]=np.squeeze(enabled_node_nums[batch_idx])
            feed_dict[pl]=temp_enabled_node_nums
        elif key=="embedded_layer":
            if not "embedded_layer" in kwargs.keys():
                chem_flag =  (info.sequence_max_length != 0) and (info.sequence_symbol_num != 0)
                if chem_flag:
                    feed_dict[pl]=np.zeros((batch_size, info.sequence_max_length,
                                            info.sequence_symbol_num))
                else:
                    feed_dict[pl]=np.zeros((batch_size, info.all_node_num,
                                            config["embedding_dim"]))
            else:
                if key in ig_targets:
                    feed_dict[pl]=kwargs["embedded_layer"]*scaling #features[batch_idx,:,:]
                else:
                    feed_dict[pl]=kwargs["embedded_layer"]

    return feed_dict
