import pandas as pd
import numpy as np
import gc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from imblearn.combine import SMOTETomek
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# 读取数据
train_1 = pd.read_csv('../raw_data/train_xy.csv')
test_1 = pd.read_csv('../raw_data/test_all.csv')
train_p_value = pd.read_csv('../generate_P_value/p_train.csv')
test_p_value = pd.read_csv('../generate_P_value/p_test.csv')
print("Load data over")

# 去除group列，取出id列
train_1 = pd.merge(train_1, train_p_value, on='cust_id')
test_1 = pd.merge(test_1, test_p_value, on='cust_id')
train1_id = train_1.pop('cust_id')
test1_id = test_1.pop('cust_id')
test_1['y'] = -1
data_1 = pd.concat([train_1, test_1], axis=0, ignore_index=True)
data_1.drop('cust_group', axis=1, inplace=True)
data_1.replace({-99 : np.nan}, inplace=True)

# 分别构建数值特征和分类特征数组
num_feature = ['x_' + str(i) for i in range(1, 96)]
cat_feature = ['x_' + str(i) for i in range(96, 158)]
# 构建20个重要特征数组
top_20_features = ['x_80', 'x_2', 'x_95', 'x_52', 'x_81', 'x_93', 'x_40', 'x_1', 'x_157', 'x_58',
                   'x_72', 'x_63', 'x_43', 'x_97', 'x_19', 'x_45', 'x_29', 'x_62', 'x_42', 'x_64']
top_20_cat = ['x_97', 'x_157']
top_20_num = []
for i in top_20_features:
    if i not in top_20_cat:
        top_20_num.append(i)
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
# 去除一个缺失值train比test多较多的离群值
data_1.drop(data_1[data_1['null_num'] > 90].index, inplace=True)

# 对于特征重要性缺失占比一定的训练集数据去除
threshold_20 = 10
data_1['null_20_num'] = data_1[top_20_features].isna().sum(axis=1)
drop_20_index = list(data_1[data_1['null_20_num'] > threshold_20].index)
drop_20_train = []
for i in drop_20_index:
    if int(i) < 15000:
        drop_20_train.append(i)
data_1.drop(drop_20_train, inplace=True)
print("Because top20 feature importance  drop %d rows" %len(drop_20_train))

# 对连续值特征中标准差小于0.1的列去除
std_df = data_1.std().reset_index()
std_df.columns = ["col_name", "std"]
low_std = std_df[std_df["std"] < 0.1]
low_std_list = low_std.col_name.tolist()
low_std_num_list = [i for i in low_std_list if i in num_feature]
data_1.drop(low_std_num_list, axis=1, inplace=True)
for item in low_std_num_list:
    if item in num_feature:
        num_feature.remove(item)
    if item in cat_feature:
        cat_feature.remove(item)
print("Because low standard deviation drop %d continuous feature" %len(low_std_num_list))

# 缺失值先填充为-99
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

train = train.values
label = label.values


# random_seed:27 dtrain_23
skf = StratifiedKFold(n_splits=3, random_state=27)
for k, (train_index, test_index) in enumerate(skf.split(train, label)):
    X_train, X_test, y_train, y_test = train[train_index], train[test_index], label[train_index], label[test_index]
    if k == 2:
        dtrain_23 = X_train
        dlabel_23 = y_train

print(dtrain_23.shape)
print(dlabel_23.shape)


X = dtrain_23
y = dlabel_23

test = test.values
del train
del label
gc.collect()

RANDOM_SEED = 1225

N = 5
skf = StratifiedKFold(n_splits=N, shuffle=False, random_state=RANDOM_SEED)

cv_result = []
pre_result = []

test_xgb = xgb.DMatrix(test, missing=-99)

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
          'subsample': 0.6,
          'min_child_weight': 8,
          'colsample_bytree': 1,
          'alpha':0.1,
          'lambda':1,
          'random_state': RANDOM_SEED,
          'silent': True,
          'nthread': -1,
          'learning_rate': 0.01,
    }

    print('*' * 20 + 'Start Round' + str(k + 1) + ' Training'+ '*' * 20)

    # train
    model_xgb = xgb.train(params,
                    xgb_train,
                    num_boost_round = 10000,
                    evals = watch_list,
                    early_stopping_rounds = 100,
                    verbose_eval = 50,
                    )

    # predict
    print('*' * 20 + 'start predict'+ '*' *20)
    for_pred = xgb.DMatrix(X_test, missing=-99)
    y_pred = model_xgb.predict(for_pred, ntree_limit=model_xgb.best_ntree_limit)
    cv_result.append(roc_auc_score(y_test, y_pred))
    print('Round ', str(k + 1),'fold AUC score is ', cv_result[k])
    pre_result.append(model_xgb.predict(test_xgb, ntree_limit=model_xgb.best_ntree_limit))
    print('Finished Round ' + str(k + 1) + '!')

five_pre = []
print('offline: cv_score: ', np.mean(cv_result))
for k, i in enumerate(pre_result):
    if k == 0:
        five_pre = np.array(i).reshape(-1,1)
    else:
        five_pre = np.hstack((five_pre, np.array(i).reshape(-1,1)))

result = []
for i in five_pre:
    result.append(np.mean(i))

sub = pd.DataFrame()
sub['cust_id'] = list(test1_id.values)
sub['pred_prob'] = list(result)

sub.to_csv('./sbumit_d23.csv', index=False, encoding='utf-8')