# Airbnb订房预测

标签（空格分隔）： 集合各模型进行预测结果对比

---

## 1、Data Exploration
###1.1、training datasets and test datasets
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
%matplotlib inline
import datetime
import os
import seaborn as sns

train = pd.read_csv("train_users_2.csv")
test = pd.read_csv("test_users.csv")

print(train.info,'\n',test.info)
##发现data_account_created是object类型，而将其转换为时间类型更方便数据分析

print(train.isnull().sum(),'\n',test.isnull().sum())
##date_first_booking和age的数据空值偏多，做数据建模时不考虑这两个特征数据；first_affiliate_tracked有3%的数据空值，需要在Feature Engineering阶段进行处理

print(train.shape,test.shape)
##训练集：(213451, 16)，测试集： (62096, 15)
```
#### Explore Each Feature
#### 1.1.1、data_account_created
```python
dac_train=train.date_account_created.value_counts()
dac_test=test.date_account_created.value_counts()
print('train.date_account_created:',
      dac_train.describe(),
      'test.date_account_created:\n',
      dac_test.describe())

dac_train_data=pd.to_datetime(dac_train.index)
dac_test_data=pd.to_datetime(dac_test.index)

dac_train_day=dac_train_data-dac_train_data.min()
dac_test_day=dac_test_data-dac_train_data.min()
##训练集选的是较早期的数据，所以减去dac_train_data.min()

print(dac_train_day,'\n',dac_test_day)
##训练集：1641 days ，测试集：91 days

plt.scatter(dac_train_day.days,dac_train.values,color='r',label='train_datasets')
plt.scatter(dac_test_day.days,dac_test.values,color='b',label='test_datasets')
plt.xlabel("Days")
plt.ylabel("The number of accounts created")
plt.title('Accounts created vs day')
##通过图像能看出训练集和测试集数据中用户数量随着时间推移
```
#### 1.1.2、timestamp_first_active
```python
tfa_train=train.timestamp_first_active.apply(lambda x: datetime.datetime.strptime(str(x),'%Y%m%d%H%M%S'))
##也可使用tfa_train=train.timestamp_first_active.astype(str).apply(lambda x: datetime.datetime.strptime(x,'%Y%m%d%H%M%S'))
```
#### 1.1.3、date_first_booking
上面已经提到空值过多，需要去除数据
#### 1.1.4、age
```python
age_train=[train.isnull().age.sum(),
           train.query('age<15').age.shape[0],
           train.query('age>=15&age<=90').age.shape[0],
           train.query('age>90').age.shape[0]]
age_test=[test.isnull().age.sum(),
          test.query('age<15').age.shape[0],
          test.query('age>=15&age<=90').age.shape[0],
          test.query('age>90').age.shape[0]]
columns=['Null','age<15','age','age>90']
print(age_train,age_test,columns)

fig,(ax1,ax2)=plt.subplots(1,2,sharex=True,sharey=True,figsize=(10,5))
sns.barplot(columns,age_train,ax=ax1)
sns.barplot(columns,age_test,ax=ax2)
ax1.set_title('training datasets')
ax2.set_title('test datasets')
ax1.set_ylabel('counts')
##通过图像观察年龄的分布
```
#### 1.1.5、Categorical features
通过图像看分类数据的分布
```python
cate_feats = ['gender', 
             'signup_method', 
             'signup_flow', 
             'language', 
             'affiliate_channel', 
             'affiliate_provider', 
             'first_affiliate_tracked', 
             'signup_app', 
             'first_device_type', 
             'first_browser']

def feature_barplot(feature, df_train = train, df_test = test, figsize=(10,5), rot = 90, saveimg = False):
    
    feat_train = df_train[feature].value_counts()
    feat_test = df_test[feature].value_counts()
    
    fig_feature, (ax1,ax2) = plt.subplots(1,2,sharex=True, sharey = True, figsize = figsize)

    sns.barplot(feat_train.index.values, feat_train.values, ax = ax1)
    sns.barplot(feat_test.index.values, feat_test.values, ax = ax2)
    
    ax1.set_xticklabels(ax1.xaxis.get_majorticklabels(), rotation = rot)
    ax2.set_xticklabels(ax1.xaxis.get_majorticklabels(), rotation = rot)

    ax1.set_title(feature + ' of train dataset')
    ax2.set_title(feature + ' of test dataset')
    ax1.set_ylabel('counts')
    
    plt.tight_layout()
    
for feat in cate_feats:
    feature_barplot(feature = feat)
```
###1.2、session datasets
```python
df_session['id']=df_session['user_id']
df_session.drop(['user_id'],axis=1)

df_session.shape

df_session.isnull().sum()

df_session.action=df_session.action.fillna('NAN')
df_session.action_type=df_session.action_type.fillna('NAN')
df_session.action_detail=df_session.action_detail.fillna('NAN')

act=dict(zip(*np.unique(df_session.action,return_counts=True)))
df_session.action=df_session.action.map(lambda x: 'OTHER' if act[x]<100 else x)
```
---
## 2、Feature Enginnering
