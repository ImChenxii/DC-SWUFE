from sklearn.cross_validation import train_test_split
import pandas as pd
import xgboost as xgb

# 读取源数据集
train = pd.read_csv('../raw_data/train_xy.csv')
test = pd.read_csv('../raw_data/test_all.csv')
# 去除cust_group列
train.drop('cust_group', axis=1, inplace=True)
test.drop('cust_group', axis=1, inplace=True)
# 取出测试集id并去除训练集id
test_id = test['cust_id']
test_x = test.drop('cust_id', axis=1)
train_x = train.drop('cust_id', axis=1)
# 取出feature，准备归为一个字典中
feature_info = {}
features = list(train_x.columns)

# 汇总训练集特征的一些统计信息
for feature in features:
    max_ = train_x[feature].max() # max value of the feature
    min_ = train_x[feature].min() # min value of the feature
    n_null = len(train_x[train_x[feature]<0])  # number of null
    n_gt1w = len(train_x[train_x[feature]>100])  # greater than 100
    feature_info[feature] = [min_,max_,n_null,n_gt1w]

print("neg:{0}, pos:{1}".format(len(train[train.y==0]), len(train[train.y==1])))


# 使用XGBoost模型来对初始训练集做特征重要性
train = train.drop(['cust_id'], axis=1)

train, val = train_test_split(train, test_size = 0.2,random_state=2018)
y = train.y
X = train.drop(['y'],axis=1)
val_y = val.y
val_X = val.drop(['y'],axis=1)


dtest = xgb.DMatrix(test_x)
dval = xgb.DMatrix(val_X,label=val_y)
dtrain = xgb.DMatrix(X, label=y)

random_seed = 2018

params={
	'booster':'gbtree',
	'objective': 'binary:logistic',
	'scale_pos_weight': float(len(y) - sum(y)) / float(sum(y)), # unblance data
    'eval_metric': 'auc',
	'max_depth':3,
    'subsample':0.65,
    'colsample_bytree':0.85,
    'min_child_weight':4,
    'eta': 0.3,
	'seed':random_seed,
	'nthread':-1,
    'learning_rate': 0.01,
    }
watchlist  = [(dtrain, 'train'),(dval, 'val')] #The early stopping is based on last set in the evallist
model = xgb.train(params, dtrain, num_boost_round=100, early_stopping_rounds=100, evals=watchlist)

print("best best_ntree_limit", model.best_ntree_limit)

test_y = model.predict(dtest,ntree_limit=model.best_ntree_limit)
test_result = pd.DataFrame(columns=["uid","score"])
test_result.uid = test_id
test_result.score = test_y

# 存储特征重要性得分和统计信息
feature_score = model.get_fscore()

for key in feature_score:
    feature_score[key] = [feature_score[key]]+feature_info[key]

feature_score = sorted(feature_score.items(), key=lambda x:x[1], reverse=True)
fs = []
for (key, value) in feature_score:
    fs.append("{0},{1},{2},{3},{4},{5}\n".format(key,value[0],value[1],value[2],value[3],value[4]))

with open('./feature_score.csv','w') as f:
    f.writelines("feature,score,min,max,n_null,n_gt1w\n")
    f.writelines(fs)