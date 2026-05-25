# 2026sta
#数科概大作业用到的代码
import pandas as pd

def explore_wine_headers(file_path):
    print("正在读取分析数据，请稍候...")

    # 使用默认逗号分隔符读取
    df = pd.read_csv(file_path, encoding='utf-8')

    # 1. 获取并展示表头清单
    columns_list = df.columns.tolist()
    print(f"\n共计识别 {len(columns_list)} 个变量，列表如下：")
    print(columns_list)

    # 2. 检查是否存在冗余 Id 列
    if 'Id' in columns_list:
        print("\n检测到 'Id' 列，该列为样本编号，与品质预测无关。")

    # 3. 检查数据是否存在缺失值
    print("\n--- 数据查询结果 ---")
    df.info()

    return df

# --- 执行入口 ---
file_path = r'E:\steam数据\WineQT.csv'
df_wine = explore_wine_headers(file_path)
#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def execute_ols_regression(file_path):
    """
    基于普通最小二乘法 (Ordinary Least Squares, OLS) 执行多元线性回归分析。
    旨在量化各项理化指标对红酒品质评分的线性边际贡献，并评估模型的整体拟合优度。
    """
    print("正在初始化普通多元线性回归计算引擎，请稍候...")

    # === 1. 数据加载与预处理 (Data Loading and Preprocessing) ===
    df = pd.read_csv(file_path, encoding='utf-8')

    # 剔除业务无关变量（如样本编号 Id）及目标因变量，构建特征矩阵 X
    X = df.drop(columns=['quality', 'Id'], errors='ignore')

    # 设定红酒品质评分为目标因变量 y
    y = df['quality']

    # === 2. 样本集划分 (Train-Test Split) ===
    # 按 8:2 比例划分训练集与测试集，固定随机状态以确保实验可复现
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # === 3. 数据标准化处理 (Data Standardization) ===
    # 消除各项理化指标间的量纲差异，将特征映射至均值为 0、方差为 1 的标准正态分布
    # 此步骤是确保回归系数具备横向可比性的必要前提
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # === 4. 模型实例化与拟合 (Model Initialization and Fitting) ===
    ols_model = LinearRegression()
    ols_model.fit(X_train_scaled, y_train)

    # === 5. 模型预测与性能评估 (Prediction and Evaluation) ===
    # 基于测试集生成预测值，计算决定系数 (R²) 以评估模型的拟合优度
    y_pred = ols_model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)

    print("\n" + "="*50)
    print("【回归模型评估报告】普通多元线性回归 (OLS)")
    print("="*50)
    print(f"模型拟合优度 (R² Score): {r2:.4f}")

    # === 6. 特征系数提取与边际贡献分析 (Coefficient Extraction) ===
    coef_df = pd.DataFrame({
        '理化指标 (Feature)': X.columns,
        '回归系数 (Coefficient)': ols_model.coef_
    })

    # 按系数绝对值降序排列，以识别核心驱动变量
    coef_df['绝对权重 (Absolute Weight)'] = coef_df['回归系数 (Coefficient)'].abs()
    coef_df = coef_df.sort_values(by='绝对权重 (Absolute Weight)', ascending=False).drop(columns=['绝对权重 (Absolute Weight)'])

    print("\n【各项理化指标回归系数分析】")
    print("说明：正数表示正向边际贡献，负数表示负向边际贡献。")
    print("-" * 50)
    print(coef_df.to_string(index=False))

    return ols_model

# --- 执行入口 ---
file_path = r'E:\steam数据\WineQT.csv'
ols_model = execute_ols_regression(file_path)
#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def execute_ols_kfold_cv(file_path, n_splits=5):
    """
    基于普通最小二乘法 (OLS) 结合 K 折交叉验证 (K-Fold CV) 执行多元线性回归分析。
    旨在消除单次数据划分的随机性，获取更稳定、严谨的模型拟合优度与边际贡献率。
    """
    print(f"正在初始化普通多元线性回归 {n_splits} 折交叉验证计算引擎，请稍候...")

    # === 1. 数据加载与预处理 (Data Loading and Preprocessing) ===
    df = pd.read_csv(file_path, encoding='utf-8')
    X = df.drop(columns=['quality', 'Id'], errors='ignore')
    y = df['quality']

    # === 2. 交叉验证器初始化 (Cross-Validator Initialization) ===
    # shuffle=True 随机化样本顺序，确保每折样本分布均匀
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    r2_scores = []
    coefs_list = []

    print(f"\n开始执行 {n_splits} 轮交叉验证...")

    # === 3. 循环执行交叉验证 (Cross-Validation Loop) ===
    for fold, (train_index, test_index) in enumerate(kf.split(X)):

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # 防泄露机制：在训练集内部拟合 Scaler，再对测试集进行转换
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        ols_model = LinearRegression()
        ols_model.fit(X_train_scaled, y_train)

        y_pred = ols_model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)

        r2_scores.append(r2)
        coefs_list.append(ols_model.coef_)

        print(f"第 {fold + 1} 轮测试 → R² 得分: {r2:.4f}")

    # === 4. 综合性能评估 (Aggregate Performance Evaluation) ===
    mean_r2 = np.mean(r2_scores)
    print("="*50)
    print(f"经过 {n_splits} 轮测试，模型平均拟合优度 (Mean R² Score): {mean_r2:.4f}")

    # === 5. 平均边际贡献分析 (Mean Coefficient Analysis) ===
    mean_coefs = np.mean(coefs_list, axis=0)

    coef_df = pd.DataFrame({
        '理化指标 (Feature)': X.columns,
        '平均回归系数 (Mean Coefficient)': mean_coefs
    })

    coef_df['绝对权重 (Absolute Weight)'] = coef_df['平均回归系数 (Mean Coefficient)'].abs()
    coef_df = coef_df.sort_values(by='绝对权重 (Absolute Weight)', ascending=False).drop(columns=['绝对权重 (Absolute Weight)'])

    print("\n【各项理化指标平均回归系数分析】")
    print("说明：正数表示正向边际贡献，负数表示负向边际贡献。此结果为多次测试的均值，更具统计学稳健性。")
    print("-" * 50)
    print(coef_df.to_string(index=False))

    return mean_r2, coef_df

# --- 执行入口 ---
file_path = r'E:\steam数据\WineQT.csv'
execute_ols_kfold_cv(file_path, n_splits=5)
#%%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

def build_wine_models(file_path):
    print("正在启动 Lasso 与随机森林回归分析，请稍候...")

    df = pd.read_csv(file_path, encoding='utf-8')
    X = df.drop(columns=['quality', 'Id'], errors='ignore')
    y = df['quality']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Lasso 回归（交叉验证自动选取最优 alpha）
    lasso = LassoCV(cv=5, random_state=42)
    lasso.fit(X_train_scaled, y_train)
    lasso_pred = lasso.predict(X_test_scaled)
    lasso_r2 = r2_score(y_test, lasso_pred)

    # 随机森林回归
    rf = RandomForestRegressor(n_estimators=45, random_state=42,max_features='sqrt')
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)
    rf_r2 = r2_score(y_test, rf_pred)

    print(f"\n模型评估结果：")
    print(f"Lasso 回归 R² 得分: {lasso_r2:.4f}（最优 alpha={lasso.alpha_:.4f}）")
    print(f"随机森林 R² 得分: {rf_r2:.4f}")

    # 随机森林特征重要性可视化
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(12, 6))

    importance_df = pd.DataFrame({
        '特征': X.columns,
        '重要性': rf.feature_importances_
    }).sort_values(by='重要性', ascending=True)

    plt.barh(importance_df['特征'], importance_df['重要性'], color='#b22222')
    plt.title('决定红酒品质的核心理化指标 (基于随机森林模型)')
    plt.xlabel('特征重要性贡献度')
    plt.tight_layout()
    plt.show()

    return rf

file_path = r'E:\steam数据\WineQT.csv'
rf_model = build_wine_models(file_path)
#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

def find_best_estimators(file_path):
    print("正在运行随机森林参数扫描，请稍候...")

    # 1. 基础数据准备
    df = pd.read_csv(file_path, encoding='utf-8')
    X = df.drop(columns=['quality', 'Id'], errors='ignore')
    y = df['quality']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 2. 设定决策树数量搜索范围，步长为 5
    estimators_range = range(10, 201, 5)
    r2_scores = []

    print("逐个测试不同数量的决策树...")

    # 3. 循环训练并记录性能
    for n in estimators_range:
        rf = RandomForestRegressor(n_estimators=n,max_features='sqrt', random_state=42)
        rf.fit(X_train_scaled, y_train)
        pred = rf.predict(X_test_scaled)

        score = r2_score(y_test, pred)
        r2_scores.append(score)
        print(f"决策树数量: {n:3d} | R² 得分: {score:.4f}")

    # 4. 定位最优参数
    best_score = max(r2_scores)
    best_n_index = r2_scores.index(best_score)
    best_n = estimators_range[best_n_index]

    print(f"\n参数扫描完成。")
    print(f"最高 R² 得分为 {best_score:.4f}，对应决策树数量为 {best_n} 棵。")

    # 5. 绘制超参数调优趋势图
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10, 5))

    sns.lineplot(x=list(estimators_range), y=r2_scores, marker='o', linewidth=2, color='#b22222')

    # 用蓝色星标标记最高点
    plt.scatter(best_n, best_score, color='blue', s=150, marker='*', zorder=5,
                label=f'最佳点 (决策树={best_n}, R²={best_score:.4f})')

    plt.title('寻找最佳决策树数量 (Hyperparameter Tuning)', fontsize=14)
    plt.xlabel('决策树的数量 (n_estimators)', fontsize=12)
    plt.ylabel('预测准确率 (R² Score)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

# --- 执行入口 ---
file_path = r'E:\steam数据\WineQT.csv'
find_best_estimators(file_path)
#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

def compare_all_regression_models(file_path):
    print("正在运行三模型横向对比，请稍候...")

    # 1. 数据加载与预处理
    df = pd.read_csv(file_path, encoding='utf-8')
    X = df.drop(columns=['quality', 'Id'], errors='ignore')
    y = df['quality']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # === 模型一：普通多元线性回归 (OLS) ===
    ols = LinearRegression()
    ols.fit(X_train_scaled, y_train)
    ols_pred = ols.predict(X_test_scaled)
    ols_r2 = r2_score(y_test, ols_pred)

    # === 模型二：Lasso 回归 (L1 正则化，自动寻优) ===
    lasso = LassoCV(cv = 5,random_state = 42)
    lasso.fit(X_train_scaled, y_train)
    lasso_pred = lasso.predict(X_test_scaled)
    lasso_r2 = r2_score(y_test, lasso_pred)

    # === 模型三：随机森林回归 (n_estimators=45，此前参数扫描确定的最优值) ===
    rf = RandomForestRegressor(n_estimators=45,max_features='sqrt', random_state=42)
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)
    rf_r2 = r2_score(y_test, rf_pred)

    print("\n" + "="*40)
    print(" 三大回归模型 R² 性能对比")
    print("="*40)
    print(f"普通多元线性回归 (OLS) 得分: {ols_r2:.4f}")
    print(f"Lasso 回归 (L1 惩罚项) 得分:   {lasso_r2:.4f}")
    print(f"随机森林回归 (非线性模型) 得分: {rf_r2:.4f}")

    print("\n" + "="*40)
    print(" 普通多元线性回归标准化系数 (边际贡献)")
    print("="*40)
    ols_coef_df = pd.DataFrame({'理化指标': X.columns, '回归系数': ols.coef_})
    ols_coef_df['影响权重(绝对值)'] = ols_coef_df['回归系数'].abs()
    ols_coef_df = ols_coef_df.sort_values(by='影响权重(绝对值)', ascending=False).drop(columns=['影响权重(绝对值)'])

    print("注：正数表示该指标越高评分越高，负数表示该指标越高评分越低。")
    print(ols_coef_df.to_string(index=False))

# --- 执行入口 ---
file_path = r'E:\steam数据\WineQT.csv'
compare_all_regression_models(file_path)
#%%
# ============================================================
# 综合改进实验：K值对比（随机森林上） + 划分比例对比
#这个单元格的结果不作使用
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, accuracy_score, f1_score, classification_report, confusion_matrix
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

file_path = r'E:\steam数据\WineQT.csv'
df = pd.read_csv(file_path, encoding='utf-8')
X = df.drop(columns=['quality', 'Id'], errors='ignore')
y = df['quality']

print("=" * 60)
print("▶ 实验一：不同 K 值的交叉验证对比 (随机森林)")
print("=" * 60)
for k in [3, 5, 10]:
    cv_scores = cross_val_score(
        RandomForestRegressor(n_estimators=100, random_state=42),
        X, y, cv=KFold(n_splits=k, shuffle=True, random_state=42),
        scoring='r2'
    )
    print(f"K={k:2d} 折 → 平均 R² = {cv_scores.mean():.4f}  (±{cv_scores.std():.4f})  |  各折: {np.round(cv_scores, 4)}")

print("\n" + "=" * 60)
print("▶ 实验二：不同训练/测试比例对比 (Random Forest)")
print("=" * 60)
for test_size in [0.1, 0.2, 0.3]:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_tr, y_tr)
    r2 = r2_score(y_te, rf.predict(X_te))
    print(f"训练:测试 = {1-test_size:.0%}:{test_size:.0%}  → 训练集={len(X_tr)}条, 测试集={len(X_te)}条  → R² = {r2:.4f}")



# ========================================================



print("\n>>> 全部实验完成！请查看上方的成绩单和图表。")
#%%
# ============================================================
# 泛化能力测试
# 注意最终放弃了这种盲检方式，因为没办法获得比较好的差异数据集
# ============================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 0. 全局配置
# ============================================================
ROUND_DECIMALS = 4          # 去重前先将数值四舍五入的精度
TEST_SIZE = 0.2             # Train/Test 划分比例
RANDOM_STATE = 42

# ============================================================
print("=" * 60)
print("▶ 阶段一：加载 WineQT 并显式划分 Train / Test")
print("=" * 60)

df_wineqt = pd.read_csv(r'E:\steam数据\WineQT.csv', encoding='utf-8')
X_all = df_wineqt.drop(columns=['quality', 'Id'], errors='ignore')
y_all = df_wineqt['quality']
feature_names = X_all.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

print(f"WineQT 总量: {len(df_wineqt)} 条")
print(f"Train Set:   {len(X_train)} 条  (用于 fit Scaler + 训练模型)")
print(f"Test Set:    {len(X_test)} 条  (仅用于 transform + 评估, 绝不参与训练)")

# ============================================================
print("\n" + "=" * 60)
print("▶ 阶段二：仅在 Train Set 上 fit Scaler，绝不接触 Test")
print("=" * 60)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)       # ← fit 仅此一次
X_test_s  = scaler.transform(X_test)            # ← transform, 不 fit

print("Scaler 已在 Train Set 上完成 fit，均值和标准差已锁定。")

# ============================================================
print("\n" + "=" * 60)
print("▶ 阶段三：仅在 Train Set 上训练三个模型")
print("=" * 60)

rf = RandomForestRegressor(n_estimators=70, random_state=RANDOM_STATE)
rf.fit(X_train_s, y_train)
rf_test_r2 = r2_score(y_test, rf.predict(X_test_s))

xgb = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=RANDOM_STATE, verbosity=0)
xgb.fit(X_train_s, y_train)
xgb_test_r2 = r2_score(y_test, xgb.predict(X_test_s))

lgb = LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8,
                     random_state=RANDOM_STATE, verbose=-1, force_row_wise=True)
lgb.fit(X_train_s, y_train)
lgb_test_r2 = r2_score(y_test, lgb.predict(X_test_s))

print(f"随机森林  → Train Set 上训练完成, Test R² = {rf_test_r2:.4f}")
print(f"XGBoost   → Train Set 上训练完成, Test R² = {xgb_test_r2:.4f}")
print(f"LightGBM  → Train Set 上训练完成, Test R² = {lgb_test_r2:.4f}")

# ============================================================
print("\n" + "=" * 60)
print("▶ 阶段四：加载新数据 + 浮点精度归一 + 去重")
print("=" * 60)

# --- 辅助函数：加载并按 4 位小数 round 所有数值列 ---
def load_and_round(path, sep, feature_cols):
    df = pd.read_csv(path, sep=sep, encoding='utf-8')
    cols_to_round = feature_cols + ['quality']
    cols_to_round = [c for c in cols_to_round if c in df.columns]
    df[cols_to_round] = df[cols_to_round].round(ROUND_DECIMALS)
    return df

# WineQT 也同样 round 一遍，保证去重基准一致
cols_round = feature_names + ['quality']
df_wineqt_rounded = df_wineqt.copy()
df_wineqt_rounded[cols_round] = df_wineqt_rounded[cols_round].round(ROUND_DECIMALS)

# 加载新红酒并 round
df_red = load_and_round(r'E:\steam数据\winequality-red.csv', ';', feature_names)

# 去重：用 round 后的 11 特征 + quality 做 merge
merge_keys = feature_names + ['quality']
red_merged = df_red.merge(df_wineqt_rounded[merge_keys], on=merge_keys,
                          how='left', indicator=True)
overlap_red = (red_merged['_merge'] == 'both').sum()
df_red_new = red_merged[red_merged['_merge'] == 'left_only'].drop(columns=['_merge'])

print(f"winequality-red.csv 原始:        {len(df_red)} 条")
print(f"与 WineQT 重叠（round 后剔除）:   {overlap_red} 条")
print(f"去重后真正陌生的样本:             {len(df_red_new)} 条")

# 加载白葡萄酒并 round
df_white = load_and_round(r'E:\steam数据\winequality-white.csv', ';', feature_names)

white_merged = df_white.merge(df_wineqt_rounded[merge_keys], on=merge_keys,
                               how='left', indicator=True)
overlap_white = (white_merged['_merge'] == 'both').sum()
df_white_new = white_merged[white_merged['_merge'] == 'left_only'].drop(columns=['_merge'])

print(f"\nwinequality-white.csv 原始:       {len(df_white)} 条")
print(f"与 WineQT 重叠（round 后剔除）:   {overlap_white} 条")
print(f"去重后真正陌生的样本:             {len(df_white_new)} 条")

# ============================================================
print("\n" + "=" * 60)
print("▶ 阶段五：对新数据仅做 transform，预测并评估")
print("=" * 60)

# --- 新红酒 ---
X_red_new = df_red_new[feature_names]
y_red_new = df_red_new['quality']
X_red_new_s = scaler.transform(X_red_new)       # 只 transform

rf_red_r2  = r2_score(y_red_new, rf.predict(X_red_new_s))
xgb_red_r2 = r2_score(y_red_new, xgb.predict(X_red_new_s))
lgb_red_r2 = r2_score(y_red_new, lgb.predict(X_red_new_s))

print("【新红酒 (去重)】")
print(f"  随机森林  R² = {rf_red_r2:.4f}")
print(f"  XGBoost   R² = {xgb_red_r2:.4f}")
print(f"  LightGBM  R² = {lgb_red_r2:.4f}")

# --- 白葡萄酒 ---
X_white_new = df_white_new[feature_names]
y_white_new = df_white_new['quality']
X_white_new_s = scaler.transform(X_white_new)   # 只 transform

rf_white_r2  = r2_score(y_white_new, rf.predict(X_white_new_s))
xgb_white_r2 = r2_score(y_white_new, xgb.predict(X_white_new_s))
lgb_white_r2 = r2_score(y_white_new, lgb.predict(X_white_new_s))

print("\n【白葡萄酒 (跨域)】")
print(f"  随机森林  R² = {rf_white_r2:.4f}")
print(f"  XGBoost   R² = {xgb_white_r2:.4f}")
print(f"  LightGBM  R² = {lgb_white_r2:.4f}")

# ============================================================
print("\n" + "=" * 60)
print("▶ 泛化能力汇总")
print("=" * 60)
print(f"{'模型':<20s} {'WineQT(Test)':>14s} {'新红酒(去重)':>14s} {'白葡萄酒(跨域)':>16s}")
print("-" * 67)
print(f"{'随机森林':<20s} {rf_test_r2:>14.4f} {rf_red_r2:>14.4f} {rf_white_r2:>16.4f}")
print(f"{'XGBoost':<20s} {xgb_test_r2:>14.4f} {xgb_red_r2:>14.4f} {xgb_white_r2:>16.4f}")
print(f"{'LightGBM':<20s} {lgb_test_r2:>14.4f} {lgb_red_r2:>14.4f} {lgb_white_r2:>16.4f}")

# ============================================================
print("\n" + "=" * 60)
print("▶ 阶段六：领域鸿沟分析 — 红酒 vs 白葡萄酒特征均值差异")
print("=" * 60)

# 用 WineQT 的 scaler 对两个数据集做标准化，便于在同一尺度上比较
X_red_all  = df_red[feature_names]
X_white_all = df_white[feature_names]

red_scaled   = pd.DataFrame(scaler.transform(X_red_all),   columns=feature_names)
white_scaled = pd.DataFrame(scaler.transform(X_white_all), columns=feature_names)

# 计算标准化后的各特征均值
mean_red   = red_scaled.mean()
mean_white = white_scaled.mean()
mean_shift = (mean_white - mean_red).abs().sort_values(ascending=True)

print("\n标准化后两域各特征的均值差异 (|Δμ| 降序排列):")
print("-" * 50)
for feat in mean_shift.index[::-1]:
    diff = mean_shift[feat]
    bar = '█' * min(int(diff * 20), 40)
    print(f"  {feat:<25s}  |Δμ| = {diff:.3f}  {bar}")

# --- 可视化：分组条形图对比红酒 vs 白葡萄酒标准化均值 ---
fig, ax = plt.subplots(figsize=(14, 7))

x_pos = np.arange(len(feature_names))
width = 0.35

bars1 = ax.bar(x_pos - width/2, mean_red.values,   width, label='红酒 (WineQT)',
               color='#b22222', edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x_pos + width/2, mean_white.values, width, label='白葡萄酒 (White)',
               color='#f4d03f', edgecolor='white', linewidth=0.5)

# 在条形上方标注差异特别大的特征
for i, feat in enumerate(feature_names):
    diff = abs(mean_red[feat] - mean_white[feat])
    if diff > 0.5:
        ax.annotate(f'Δ={diff:.2f}', (x_pos[i], max(mean_red[feat], mean_white[feat]) + 0.1),
                    ha='center', fontsize=8, color='#c0392b', fontweight='bold')

ax.set_xticks(x_pos)
ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('标准化后的特征均值 (以 WineQT Scaler 为基准)', fontsize=11)
ax.set_title('红酒 vs 白葡萄酒：标准化特征均值对比\n(差距越大, 跨域预测越困难)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# 打印一句总括分析
top3_shift = mean_shift[::-1][:3]
print(f"\n差异最大的三个特征: {', '.join(top3_shift.index)}")
print("正是这些理化指标在两个域间的系统性偏移，解释了红酒模型在白酒上近乎失效的原因。")

print("\n>>> 全部泛化测试及鸿沟分析完成。")
