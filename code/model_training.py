import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

print("开始构造标签和训练模型...")

# 第一步：读取数据

print("\n[1/7] 读取特征数据...")
user_features = pd.read_csv('user_features.csv')
print(f"用户特征数：{len(user_features)}")

# 读取原始数据
df = pd.read_csv('user_behavior_cleaned_sample.csv')
df['time'] = pd.to_datetime(df['time'])
print(f"原始数据行数：{len(df):,}")

# 查看数据列名，确认字段
print(f"\n数据列名：{df.columns.tolist()}")


# 第二步：构造标签（12月19日是否有购买）

print("\n[2/7] 构造预测标签...")

# 定义预测日
target_date = pd.Timestamp('2014-12-19')

# 先查看一下12月19日是否有数据
print("\n12月19日数据量：")
print(df[df['time'].dt.date == target_date.date()].shape)

# 查看行为类型分布
print("\n行为类型分布：")
print(df['behavior_type'].value_counts())

# 找出12月19日有购买行为的用户
buy_12_19 = df[
    (df['time'].dt.date == target_date.date()) &
    (df['behavior_type'] == 4)
    ]['user_id'].unique()

print(f"\n12月19日有购买行为的用户数：{len(buy_12_19)}")

# 如果12月19日没有购买数据，改用最后一天的数据
if len(buy_12_19) == 0:
    print("\n⚠️ 12月19日无购买数据，改用12月18日作为预测日...")

    # 找到数据最后一天
    last_date = df['time'].dt.date.max()
    print(f"数据最后一天：{last_date}")

    # 用最后一天作为预测日
    buy_last_day = df[
        (df['time'].dt.date == last_date) &
        (df['behavior_type'] == 4)
        ]['user_id'].unique()

    print(f"{last_date} 有购买行为的用户数：{len(buy_last_day)}")

    # 构造标签
    user_features['label'] = user_features['user_id'].isin(buy_last_day).astype(int)
else:
    # 构造标签
    user_features['label'] = user_features['user_id'].isin(buy_12_19).astype(int)

# 检查标签分布
label_dist = user_features['label'].value_counts()
print(f"\n标签分布：")
print(f"正样本（有购买）：{label_dist.get(1, 0)} 人 ({label_dist.get(1, 0) / len(user_features) * 100:.2f}%)")
print(f"负样本（无购买）：{label_dist.get(0, 0)} 人 ({label_dist.get(0, 0) / len(user_features) * 100:.2f}%)")

# 如果没有正样本，退出
if label_dist.get(1, 0) == 0:
    print("\n❌ 错误：没有正样本，无法训练！")
    print("建议：使用更早的日期作为预测日，或者换一个数据集。")
    exit()


# 第三步：准备训练数据

print("\n[3/7] 准备训练数据...")

# 特征列（排除user_id和label）
feature_cols = [col for col in user_features.columns if col not in ['user_id', 'label']]
X = user_features[feature_cols]
y = user_features['label']

print(f"特征维度：{X.shape}")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"训练集大小：{len(X_train)}")
print(f"测试集大小：{len(X_test)}")

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 第四步：训练多个模型

print("\n[4/7] 开始训练模型...")

results = {}

# 1. 逻辑回归
print("\n训练逻辑回归...")
lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict_proba(X_test_scaled)[:, 1]
auc_lr = roc_auc_score(y_test, y_pred_lr)
results['Logistic Regression'] = auc_lr
print(f"逻辑回归 AUC: {auc_lr:.4f}")

# 2. 随机森林
print("\n训练随机森林...")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict_proba(X_test)[:, 1]
auc_rf = roc_auc_score(y_test, y_pred_rf)
results['Random Forest'] = auc_rf
print(f"随机森林 AUC: {auc_rf:.4f}")

# 3. XGBoost
print("\n训练XGBoost...")
# 计算正负样本比例
scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
xgb = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict_proba(X_test)[:, 1]
auc_xgb = roc_auc_score(y_test, y_pred_xgb)
results['XGBoost'] = auc_xgb
print(f"XGBoost AUC: {auc_xgb:.4f}")


# 第五步：结果对比

print("\n[5/7] 模型效果对比：")
print("-" * 40)
for model, auc in results.items():
    print(f"{model}: AUC = {auc:.4f}")
print("-" * 40)

best_model = max(results, key=results.get)
print(f"\n最佳模型：{best_model}，AUC = {results[best_model]:.4f}")


# 第六步：特征重要性分析

print("\n[6/7] 特征重要性分析...")

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 重要特征：")
print(feature_importance.head(10))

# 可视化
plt.figure(figsize=(10, 6))
plt.barh(feature_importance.head(10)['feature'], feature_importance.head(10)['importance'])
plt.xlabel('重要性')
plt.title('Top 10 特征重要性')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=100)
plt.show()
print("\n特征重要性图已保存为 feature_importance.png")


# 第七步：保存结果

print("\n[7/7] 保存结果...")

# 保存最佳模型
import joblib

if best_model == 'Logistic Regression':
    joblib.dump(lr, 'best_model_lr.pkl')
elif best_model == 'Random Forest':
    joblib.dump(rf, 'best_model_rf.pkl')
else:
    joblib.dump(xgb, 'best_model_xgb.pkl')

# 保存结果汇总
results_df = pd.DataFrame(list(results.items()), columns=['Model', 'AUC'])
results_df.to_csv('model_results.csv', index=False)

print("\n结果已保存：")
print("- best_model_*.pkl（最佳模型）")
print("- model_results.csv（模型效果对比）")
print("- feature_importance.png（特征重要性图）")

# 最终报告
print("\n" + "=" * 60)
print(f"项目总结：")
print(f"- 用户数：{len(user_features)}")
print(f"- 特征数：{len(feature_cols)}")
print(f"- 正样本比例：{label_dist.get(1, 0) / len(user_features) * 100:.2f}%")
print(f"- 最佳模型：{best_model}")
print(f"- 最佳AUC：{results[best_model]:.4f}")
print("=" * 60)
