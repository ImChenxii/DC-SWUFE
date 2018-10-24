import pandas as pd

top_all = pd.read_csv('./top_all.csv')
top_all_np = pd.read_csv('./top_all_np.csv')
top_80 = pd.read_csv('./sbumit_feature_top80.csv')# 0.870768(offline)
top_80_np = pd.read_csv('./sbumit_feature_top80_np.csv')# 0.8078460321634381(offline)
top_50 = pd.read_csv('./sbumit_feature_top50.csv')# 0.869899(offline)
top_50_np = pd.read_csv('./sbumit_feature_top50_np.csv')# 0.8161498184241698(offline)

fea_15_15_1 = pd.read_csv('./sbumit_feature_15_15_1.csv')# 0.8694(offline)
fea_15_15_1_np = pd.read_csv('./sbumit_feature_15_15_1_np.csv')# 0.8163923977531946(offline)
fea_15_15_2 = pd.read_csv('./sbumit_feature_15_15_2.csv')# 0.8757(offline)
fea_15_15_2_np = pd.read_csv('./sbumit_feature_15_15_2_np.csv')# 0.824612708491237(offline)

cust_id = top_all.cust_id

weight = [0.3, 0.07, 0.08, 0.15, 0.15, 0.05, 0.05, 0.05, 0.05, 0.05]
sum = 0
for i in weight:
    sum += i
print(sum)

pred_prob = weight[0] * top_all.pred_prob + weight[1] * top_80.pred_prob + \
            weight[2] * top_50.pred_prob + weight[3] * fea_15_15_1.pred_prob + \
            weight[4] * fea_15_15_2.pred_prob + weight[5] * top_all_np.pred_prob + \
            weight[6] * top_50_np.pred_prob + weight[7] * top_80_np.pred_prob + \
            weight[8] * fea_15_15_1_np.pred_prob + weight[9] * fea_15_15_2_np.pred_prob

pred = pd.DataFrame(cust_id, columns=['cust_id'])
pred['pred_prob'] = pred_prob

pred.to_csv('./M3.csv', index=None, encoding='utf-8')