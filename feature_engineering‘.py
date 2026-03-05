# 特征工程
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

print("开始特征工程...")

# 1. 读取清洗后的数据
df = pd.read_csv('user_behavior_cleaned_sample.csv')
print(f"读取数据：{len(df):,} 行")

# 2. 转换时间格式
df['time'] = pd.to_datetime(df['time'])
df['date'] = df['time'].dt.date
df['hour'] = df['time'].dt.hour

# 3. 定义训练期和预测期
# 训练期：11.18-12.18
# 预测日：12.19

print("\n开始构造特征（按用户聚合）...")
# ============================================
# 特征组1：用户整体活跃度特征
# ============================================
print("\n[1/6] 构造用户活跃度特征...")

user_features = pd.DataFrame()
user_features['user_id'] = df['user_id'].unique()

# 总行为数
user_total = df.groupby('user_id').size().reset_index(name='total_actions')

# 各行为计数
behavior_counts = df.groupby(['user_id', 'behavior_type']).size().unstack(fill_value=0)
behavior_counts.columns = ['view_cnt', 'fav_cnt', 'cart_cnt', 'buy_cnt']

# 合并
user_features = user_features.merge(user_total, on='user_id', how='left')
user_features = user_features.merge(behavior_counts, on='user_id', how='left')

# 填充缺失（如果某个用户没有某种行为）
user_features = user_features.fillna(0)

print(f"已完成，当前特征数：{len(user_features.columns)}")

# ============================================
# 特征组2：转化率特征（核心！）
# ============================================
print("\n[2/6] 构造转化率特征...")

# 浏览→购买转化率
user_features['view_to_buy_rate'] = user_features['buy_cnt'] / (user_features['view_cnt'] + 1)

# 收藏→购买转化率
user_features['fav_to_buy_rate'] = user_features['buy_cnt'] / (user_features['fav_cnt'] + 1)

# 加购→购买转化率
user_features['cart_to_buy_rate'] = user_features['buy_cnt'] / (user_features['cart_cnt'] + 1)

# 购买占比（用户所有行为中购买的比例）
user_features['buy_ratio'] = user_features['buy_cnt'] / user_features['total_actions']

print("转化率特征构造完成")

# ============================================
# 特征组3：时间特征
# ============================================
print("\n[3/6] 构造时间特征...")

# 用户最后一次行为时间
last_time = df.groupby('user_id')['time'].max().reset_index()
last_time['time'] = pd.to_datetime(last_time['time'])
# 计算距12.19的天数（预测日是12.19）
target_date = pd.Timestamp('2014-12-19')
last_time['days_since_last_action'] = (target_date - last_time['time']).dt.days
last_time = last_time[['user_id', 'days_since_last_action']]

user_features = user_features.merge(last_time, on='user_id', how='left')

# 用户活跃天数
active_days = df.groupby('user_id')['date'].nunique().reset_index(name='active_days')
user_features = user_features.merge(active_days, on='user_id', how='left')

print("时间特征构造完成")

# ============================================
# 特征组4：商品类目偏好
# ============================================
print("\n[4/6] 构造商品类目偏好特征...")

# 用户浏览过的类目数
user_categories = df[df['behavior_type'] == 1].groupby('user_id')['item_category'].nunique().reset_index(name='view_categories')
user_features = user_features.merge(user_categories, on='user_id', how='left').fillna(0)

# 用户购买过的类目数
user_buy_categories = df[df['behavior_type'] == 4].groupby('user_id')['item_category'].nunique().reset_index(name='buy_categories')
user_features = user_features.merge(user_buy_categories, on='user_id', how='left').fillna(0)

print("类目特征构造完成")

# ============================================
# 特征组5：用户行为强度特征
# ============================================
print("\n[5/6] 构造行为强度特征...")

# 平均每天行为数
user_features['avg_daily_actions'] = user_features['total_actions'] / user_features['active_days']

# 非购买行为占比（浏览+收藏+加购）/总行为
user_features['non_buy_ratio'] = (user_features['view_cnt'] + user_features['fav_cnt'] + user_features['cart_cnt']) / user_features['total_actions']

# 购物车深度（加购/购买）
user_features['cart_depth'] = user_features['cart_cnt'] / (user_features['buy_cnt'] + 1)

print("强度特征构造完成")

# ============================================
# 特征组6：构造标签（这一步是为第4-5天准备）
# ============================================
print("\n[6/6] 准备标签数据...")

# 注意：这里只是演示特征构造，真正的标签需要另外处理
# 完整标签会在第4-5天代码中提供

# 查看特征概览
print("\n" + "="*60)
print("特征构造完成！特征维度：")
print(f"用户数：{len(user_features)}")
print(f"特征数：{len(user_features.columns)}")
print("\n特征列名：")
print(user_features.columns.tolist())

# 显示前几行
print("\n特征表示例（前5行）：")
print(user_features.head())

# 保存特征文件
user_features.to_csv('user_features.csv', index=False)
print("\n✅ 特征已保存到 user_features.csv")

# 特征统计描述
print("\n特征统计描述：")
print(user_features.describe())

# 检查是否有异常值
print("\n检查异常值（负值、无穷大）：")
print(user_features.select_dtypes(include=[np.number]).lt(0).sum())
print("\n检查无穷大：")
print(np.isinf(user_features.select_dtypes(include=[np.number])).sum())

print("\n" + "="*60)
