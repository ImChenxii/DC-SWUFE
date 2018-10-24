import pandas as pd
import numpy as np
import gc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# 读取数据
train_1 = pd.read_csv('../raw_data/train_xy.csv')
test_1 = pd.read_csv('.../raw_data/test_all.csv')
print("Load data over")

# 去除group列，取出id列
train1_id = train_1.pop('cust_id')
test1_id = test_1.pop('cust_id')
test_1['y'] = -1
data_1 = pd.concat([train_1, test_1], axis=0, ignore_index=True)
data_1.drop('cust_group', axis=1, inplace=True)
data_1.replace({-99 : np.nan}, inplace=True)

# 分别构建数值特征和分类特征数组
num_feature = ['x_' + str(i) for i in range(1, 96)]
cat_feature = ['x_' + str(i) for i in range(96, 158)]

# 去除常量features
unique_df = data_1.nunique().reset_index()
unique_df.columns = ["col_name", "unique_count"]
constant_df = unique_df[unique_df["unique_count"] == 1]
constant_feature = constant_df.col_name.tolist()
data_1.drop(constant_feature, axis=1, inplace=True)
for item in constant_feature:
    if item in num_feature:
        num_feature.remove(item)
    if item in cat_feature:
        cat_feature.remove(item)
print("drop ", len(constant_feature), " constant feature(s)")

# 去除缺失值占比大于80%的feature
mis_80_list = []
for col in data_1.columns:
    mis_val_percent = 100 * data_1[col].isnull().sum() / len(data_1)
    if (mis_val_percent >= 80.0):
        mis_80_list.append(col)
data_1.drop(mis_80_list, axis=1, inplace=True)
for item in mis_80_list:
    if item in num_feature:
        num_feature.remove(item)
    if item in cat_feature:
        cat_feature.remove(item)
print("drop ", len(mis_80_list), "missing feature(s)")

# 缺失值个数当做一个特征来统计用户信息完整度
data_1['null_num'] = data_1.isna().sum(axis=1)

# 对连续值特征中标准差小于0.1的行去除
std_df = data_1.std().reset_index()
std_df.columns = ["col_name", "std"]
low_std = std_df[std_df["std"] < 0.1]
low_std_list = low_std.col_name.tolist()
low_std_num_list = [i for i in low_std_list if i in num_feature]
print("Because low standard deviation drop %d continuous feature" %len(low_std_num_list))

# 缺失值填充为-99
data_1.fillna(-99, inplace=True)

gc.collect()

# 对类别特征进行One-hot和LabelEncoder
unique_df = data_1.nunique().reset_index()
unique_df.columns = ["col_name", "unique_count"]
category2_df = unique_df[unique_df["unique_count"] == 2]
category2_feature = category2_df.col_name.tolist()

if 'y' in category2_feature:
    category2_feature.remove('y')

le = LabelEncoder()
for col in category2_feature:
    le.fit(data_1[col])
    data_1[col] = le.transform(data_1[col])

for item in category2_feature:
    if item in cat_feature:
        cat_feature.remove(item)

def one_hot_encode(data, column_name):
    dummies = pd.get_dummies(data[column_name], prefix=column_name)
    combined = data.join(dummies)
    combined.drop(column_name, axis=1, inplace=True)
    return combined
for col_name in cat_feature:
    data_1 = one_hot_encode(data_1, col_name)
    print(col_name, " one-hot is over.")

train = data_1[(data_1['y'] != -1) & (data_1['y'] != -2)]
test = data_1[data_1['y'] == -1]
label = train.pop('y')
test.drop('y', axis=1, inplace=True)
del data_1
gc.collect()

print("train shape is ", train.shape)
print("test shape is", test.shape)

X = train.values
y = label.values

test = test.values
del train
del label
gc.collect()

# 使用5折交叉验证
N = 5
skf = StratifiedKFold(n_splits=N, shuffle=False, random_state=2018)

cv_result = []
pre_result = []
pre_train_result = []

test_xgb = xgb.DMatrix(test, missing=-99)
train_p_xgb = xgb.DMatrix(X, missing=-99)

for k, (train_index, test_index) in enumerate(skf.split(X, y)):
    print('*' * 20 + 'Start Round ' + str(k + 1) + ' Split' + '*' * 20)
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    # create dataset for xgboost
    xgb_train = xgb.DMatrix(X_train, y_train, missing=-99)
    xgb_eval = xgb.DMatrix(X_test, y_test, missing=-99)
    watch_list = [(xgb_train, 'train'), [xgb_eval, 'eval']]
    # specify your configurations as a dict
    params = {
          'booster': 'gbtree',
          'objective': 'binary:logistic',
          'eval_metric': 'auc',
          'eta': 0.3,
          'max_depth': 3,
          'subsample': 0.65,
          'min_child_weight': 4,
          'colsample_bytree': 0.85,
          'alpha':0,
          'random_state': 2018,
          'silent': True,
          'nthread': -1,
          'learning_rate': 0.01,
    }

    print('*' * 20 + 'Start Round' + str(k + 1) + ' Training'+ '*' * 20)

    # train
    model_xgb = xgb.train(params,
                    xgb_train,
                    num_boost_round = 2000,
                    evals = watch_list,
                    early_stopping_rounds = 50,
                    verbose_eval = 50,
                    )

    # predict
    print('*' * 20 + 'start predict'+ '*' *20)
    for_pred = xgb.DMatrix(X_test, missing=-99)
    y_pred = model_xgb.predict(for_pred, ntree_limit=model_xgb.best_ntree_limit)
    cv_result.append(roc_auc_score(y_test, y_pred, reorder=True))
    print('Round ', str(k + 1),'fold AUC score is ', cv_result[k])
    pre_result.append(model_xgb.predict(test_xgb, ntree_limit=model_xgb.best_ntree_limit))
    pre_train_result.append(model_xgb.predict(train_p_xgb, ntree_limit=model_xgb.best_ntree_limit))
    print('Finished Round ' + str(k + 1) + '!')

five_pre = []
five_train_pre = []
print('offline: cv_score: ', np.mean(cv_result))
for k, i in enumerate(pre_result):
    if k == 0:
        five_pre = np.array(i).reshape(-1,1)
    else:
        five_pre = np.hstack((five_pre, np.array(i).reshape(-1,1)))

for k, i in enumerate(pre_train_result):
    if k == 0:
        five_train_pre = np.array(i).reshape(-1,1)
    else:
        five_train_pre = np.hstack((five_train_pre, np.array(i).reshape(-1,1)))

result = []
result_train = []
for i in five_pre:
    result.append(np.mean(i))

for i in five_train_pre:
    result_train.append(np.mean(i))

sub = pd.DataFrame()
sub_p = pd.DataFrame()
sub['cust_id'] = list(test1_id.values)
sub['p_value'] = list(result)

sub_p['cust_id'] = list(train1_id.values)
sub_p['p_value'] = list(result_train)
# 生成p值文件
sub.to_csv('./p_test.csv', index=False, encoding='utf-8')
sub_p.to_csv('./p_train.csv', index=False, encoding='utf-8')