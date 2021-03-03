#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import *


if __name__ == "__main__":


    data = pd.read_csv('./beijing_cate2id.csv')
   
#     sparse_features = ["whois_blk_regist_netname", 'whois_blk_regist_orgname']
    sparse_features = ["whois_blk_regist_netname", 'whois_blk_regist_orgname', 'blk_province', 'blk_city']
    dense_features = ['cover_ip_num', 'blk_landmark_num','blk_landmark_ratio','avg_his_ll_num','blk_landmark_cover_area_radius','blk_landmark_cover_province_num','blk_landmark_cover_city_num','blk_landmark_cover_district_num','avg_ip_his_ll_cover_area_ratio','icmp_alive_ip_num','icmp_alive_ip_ratio','p80_alive_ip_num','p80_alive_ip_ratio','p443_alive_ip_num','p443_alive_ip_ratio','p21_alive_ip_num','p21_alive_ip_ratio','p22_alive_ip_num','p22_alive_ip_ratio','p23_alive_ip_num','p23_alive_ip_ratio','p53_alive_ip_num','p53_alive_ip_ratio','email_port_alive_ip_num','email_port_alive_ip_ratio','web_banner_ip_num','web_banner_ip_ratio','enterprise_router_banner_ip_ratio','web_banner_soft_version_num','tr_reachable_ip_num','tr_reachable_ip_ratio','tr_router_ip_num','tr_router_ip_ratio','avg_main_domain_num','has_domain_ip_num','has_domain_ip_ratio','whois_blk_ip_num','bgp_prefix_sub_num']
    
    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    target = ['scene']
    
    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                              for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                              for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)

    # 3.generate input data for model

    train, test = train_test_split(data, train_size=0.75, shuffle=True, random_state=1023)
    train_model_input = {name: train[name] for name in feature_names}
    test_model_input = {name: test[name] for name in feature_names}

    # 4.Define Model,train,predict and evaluate

    device = 'cpu'
    # if occurs runtime error, just use CPU instead of GPU.
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

# 3. D&CN
    model = DCN(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,task="multiclass",l2_reg_embedding=1e-5, device=device)
# 4. AutoInt
#     model = AutoInt(linear_feature_columns=linear_feature_columns, dnn_feature_columns=dnn_feature_columns,task="multiclass",l2_reg_embedding=1e-5, device=device)

    
    model.compile("adam", "cross_entropy",
                  metrics=["auc"], )
    model.fit(train_model_input, train[target].values,
              batch_size=1000, epochs=50, validation_split=0.0, verbose=0)

    pred_ans = model.predict(test_model_input, 512)
    
    train_ans =  model.predict(train_model_input, 512)
    print("")
    print("train AUC", round(roc_auc_score(train[target].values, train_ans, multi_class="ovr", average='macro', labels=[0, 1, 2, 3]), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans, multi_class="ovr", average='macro', labels=[0, 1, 2, 3]), 4))

