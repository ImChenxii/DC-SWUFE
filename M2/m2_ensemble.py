import pandas as pd


d_all = pd.read_csv('./dtrain_all.csv')

d_12 = pd.read_csv('./sbumit_d12.csv')
d_12_SMOTE = pd.read_csv('./sbumit_d12_SMOTE.csv')

d_13 = pd.read_csv('./sbumit_d13.csv')
d_13_SMOTE = pd.read_csv('./sbumit_d13_SMOTE.csv')

d_23 = pd.read_csv('./sbumit_d23.csv')
d_23_SMOTE = pd.read_csv('./sbumit_d23_SMOTE.csv')

cust_id = d_all.cust_id
weight = [0.6, 0.07, 0.03, 0.1, 0.07, 0.08, 0.05]

sum = sum(weight)
print(sum)

pred_prob = weight[0] * d_all.pred_prob + weight[1] * d_12.pred_prob + weight[2] * d_12_SMOTE.pred_prob + \
            weight[3] * d_13.pred_prob + weight[4] * d_13_SMOTE.pred_prob + weight[5] * d_23.pred_prob +\
            weight[6] * d_23_SMOTE.pred_prob

pred = pd.DataFrame(cust_id, columns=['cust_id'])
pred['pred_prob'] = pred_prob

pred.to_csv('./M2.csv', index=None, encoding='utf-8')