import pandas as pd

M2 = pd.read_csv('./M2/M2.csv')
M3 = pd.read_csv('./M3/M3.csv')
M4 = pd.read_csv('./M4/M4.csv')

cust_id = M4.cust_id
weight_m2 = 0
weight_m3 = 0.45
weight_m4 = 0.55

pred_prob = weight_m2 * M2.pred_prob + weight_m3 * M3.pred_prob + weight_m4 * M4.pred_prob

pred = pd.DataFrame(cust_id, columns=['cust_id'])
pred['pred_prob'] = pred_prob

pred.to_csv('./submit.csv', index=None, encoding='utf-8')