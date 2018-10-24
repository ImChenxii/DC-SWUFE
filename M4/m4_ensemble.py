import pandas as pd

xgb_p = pd.read_csv('./xgb_p.csv')
xgb_un_p = pd.read_csv('./xgb_un_p.csv')

lgb_p = pd.read_csv('./lgb_p.csv')
lgb_un_p = pd.read_csv('./lgb_un_p.csv')

cust_id = xgb_p.cust_id
weight = [0.45, 0.15, 0.3, 0.1]

sum = 0
for i in weight:
    sum += i
print(sum)

pred_prob = weight[0] * xgb_p.pred_prob + weight[1] * xgb_un_p.pred_prob + \
            weight[2] * lgb_p.pred_prob + weight[3] * lgb_un_p.pred_prob

pred = pd.DataFrame(cust_id, columns=['cust_id'])
pred['pred_prob'] = pred_prob

pred.to_csv('./M4.csv', index=None, encoding='utf-8')