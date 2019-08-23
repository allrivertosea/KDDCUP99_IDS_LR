# -*- coding: utf-8 -*
import pandas as pd
from sklearn.preprocessing import  MinMaxScaler,StandardScaler
from sklearn.cluster import KMeans
def mapfeatonum2(df):#将正常映射为0，攻击映射为1，进行二分类
    d = {'normal.': 0,'ipsweep.': 1,'mscan.': 1,'nmap.': 1,'portsweep.': 1,'saint.': 1,'satan.': 1,'apache2.': 1,
        'back.': 1,'mailbomb.': 1,'neptune.': 1,'pod.': 1,'land.': 1,'processtable.': 1,'smurf.': 1,'teardrop.': 1,'udpstorm.': 1,
        'buffer_overflow.': 1,'loadmodule.': 1,'perl.': 1,'ps.': 1,'rootkit.': 1,'sqlattack.': 1,'xterm.': 1,
        'ftp_write.': 1,'guess_passwd.': 1, 'httptunnel.': 1,  'imap.': 1,'multihop.': 1,  'named.': 1,'phf.': 1,
        'sendmail.': 1, 'snmpgetattack.': 1,'snmpguess.': 1,'worm.': 1, 'xlock.': 1,'xsnoop.': 1, 'spy.': 1,
        'warezclient.': 1, 'warezmaster.': 1 }
    l = []
    for val in df['label']:
        l.append(d[val])
    tmp_df = pd.DataFrame(l, columns=['label'])
    df = df.drop('label', axis=1)
    df = df.join(tmp_df)
    return df

def merge_sparse_feature(df):#合并稀疏特征
    df.loc[(df['service'] == 'ntp_u')
           | (df['service'] == 'urh_i')
           | (df['service'] == 'tftp_u')
           | (df['service'] == 'red_i')
    , 'service'] = 'normal_service_group'

    df.loc[(df['service'] == 'pm_dump')
           | (df['service'] == 'http_2784')
           | (df['service'] == 'harvest')
           | (df['service'] == 'aol')
           | (df['service'] == 'http_8001')
    , 'service'] = 'satan_service_group'
    return df


def data_process(data_train):
    #第一步：类标转换，稀疏特征合并
    data_train = mapfeatonum2(data_train)
    data_train = merge_sparse_feature(data_train)
    #第二步：连输属性离散化处理
    con_columns = ['duration','src_bytes','dst_bytes','wrong_fragment','urgent','hot','num_failed_logins',
                          'num_compromised','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds',
                          'count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate',
                          'diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate',
                          'dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate',
                          'dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate']
    dis_columns = ['protocol_type','service','flag','land','logged_in','root_shell','su_attempted','is_host_login','is_guest_login','label']
    con_fea = data_train[con_columns]
    dis_fea = data_train[dis_columns]
#-----------------------------------
#利用聚类离散化，聚类算法使用sklearn的
#-----------------------------------
    con_fea_new = con_fea.copy()
    for i in con_columns:
        if len(con_fea[i].unique())>=10:
            k = 10
            kmodel = KMeans(n_clusters=k)
        else:
            k = len(con_fea[i].unique())
            kmodel = KMeans(n_clusters=k)
        kmodel.fit(con_fea[i].values.reshape(-1, 1))  # 训练模型
        cls_ce = pd.DataFrame(kmodel.cluster_centers_, columns=list('a')).sort_values(by='a')
        bin = cls_ce.rolling(window=2).mean()['a'][1:].tolist()
        bin = [-1] + bin + [con_fea[i].values.max()+1]
        feature_val = con_fea[i].tolist()#series转为list
        new_feature_val = pd.cut(feature_val, bin, labels=range(k))
        con_fea_new[i] = new_feature_val
#--------------------------------------
#合并：将con_fea_new与dis_fea合并为data_train_new（这之前先删除num_outbound_cmds）
#--------------------------------------
    con_fea_new.drop(['num_outbound_cmds'],axis =1,inplace=True)
    data_train_new = con_fea_new.join(dis_fea)#将离散属性添加上
#--------------------------------------
#独热编码：处理标称离散特征,只处理3个即可
#--------------------------------------
    data_encoded = pd.get_dummies(data_train_new, columns=['protocol_type', 'service', 'flag'])#独热编码
    col_name = data_encoded.columns.tolist()
    col_name = col_name[0:1]+col_name[38:]+col_name[1:38]#将独热编码后的属性放在原来的位置，将标签放在最后
    data_train_new = data_encoded.reindex(columns=col_name)#重新索引columns,构建data_train_new
    # data_train_new.to_csv(r'E:\入侵检测\机器学习在入侵检测的应用\逻辑回归\dataset\最终数据.csv')
#---------------------------------------
#全体数据标准化，归一化
#---------------------------------------
    std_scaler = StandardScaler()
    data_train_new_std = std_scaler.fit_transform(data_train_new.values)
    min_max_scaler = MinMaxScaler()
    data_train_new_norm = min_max_scaler.fit_transform(data_train_new_std)
    data_train_new_norm = pd.DataFrame(data_train_new_norm,columns=data_train_new.columns)
    return data_train_new_norm

if __name__=="__main__":
    file_train = r'E:\入侵检测\机器学习在入侵检测的应用\逻辑回归\dataset\kddcup.data_10_percent_corrected.csv'
    file_test = r'E:\入侵检测\机器学习在入侵检测的应用\逻辑回归\dataset\corrected.csv'
    data_train = pd.read_csv(file_train)
    data_test = pd.read_csv(file_test)
    data_train_new_norm = data_process(data_train)
    data_test_new_norm = data_process(data_test)
    data_train_new_norm.to_csv(r'E:\入侵检测\机器学习在入侵检测的应用\逻辑回归\dataset\data_train_new_norm.csv')
    data_test_new_norm.to_csv(r'E:\入侵检测\机器学习在入侵检测的应用\逻辑回归\dataset\data_test_new_norm.csv')

