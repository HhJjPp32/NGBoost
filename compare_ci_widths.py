#!/usr/bin/env python3
"""
可视化对比 95% 与 99% 置信区间的区间宽度。
读取两个已保存的 NGBoost 模型，在相同测试集上计算预测区间并绘制对比图。
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data_loader import DataLoader
from src.preprocessor import DataPreprocessor
from src.model_trainer import XGBoostTrainer
from src.ngboost_trainer import NGBoostTrainer
from src.utils import load_config, setup_logger

# ------------------------------------------------------------------
# 配置路径
# ------------------------------------------------------------------
CONFIG_PATH = "config/config_ngboost.yaml"
XGB_MODEL_PATH = "output/3.16_150rounds/xgboost_model.pkl"

MODEL_95_PATH = "output/ngboost_20260415_122527/ngboost_residual_model.pkl"
MODEL_99_PATH = "output/ngboost_20260415_140029/ngboost_residual_model.pkl"

OUTPUT_DIR = Path("output/ci_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# 加载数据与 XGBoost 基模型
# ------------------------------------------------------------------
config = load_config(CONFIG_PATH)
logger = setup_logger("ci_compare", level="INFO")

logger.info("Loading data...")
data_loader = DataLoader(config, logger)
df, feature_cols = data_loader.load_and_prepare(config['paths']['raw_data'])

preprocessor = DataPreprocessor(config, logger)
df_processed, feature_cols = preprocessor.preprocess(df, feature_cols, fit=True)

X_train, X_test, y_train, y_test = preprocessor.split_data(
    df_processed, feature_cols, apply_log_transform=True
)

use_log_transform = config['preprocessing'].get('log_transform_target', False)

logger.info("Loading XGBoost model...")
xgb_trainer = XGBoostTrainer(config, logger)
xgb_trainer.load_model(XGB_MODEL_PATH)

# XGBoost 点预测（原始尺度）
y_test_pred_log = xgb_trainer.predict(X_test)
if use_log_transform:
    y_test_pred = preprocessor.inverse_transform_target(y_test_pred_log)
    y_test_orig = preprocessor.inverse_transform_target(y_test)
else:
    y_test_pred = y_test_pred_log
    y_test_orig = y_test

# XGBoost 方差特征
logger.info("Calculating XGBoost prediction variance...")
n_estimators = xgb_trainer.model.n_estimators
all_predictions = []
for i in range(n_estimators):
    pred = xgb_trainer.model.predict(X_test, iteration_range=(i, i + 1))
    if use_log_transform:
        pred = preprocessor.inverse_transform_target(pred)
    all_predictions.append(pred)
var_test = np.var(np.array(all_predictions), axis=0)

if X_test.ndim == 1:
    X_test = X_test.reshape(-1, 1)
X_test_with_var = np.column_stack([X_test, var_test])

# ------------------------------------------------------------------
# 辅助函数：加载 NGBoost 并计算区间
# ------------------------------------------------------------------
def load_ngboost_and_intervals(model_path, confidence):
    ng_trainer = NGBoostTrainer(config, logger)
    ng_trainer.load_model(model_path)

    # 残差区间
    eps_lower, eps_upper = ng_trainer.predict_interval(X_test_with_var, confidence=confidence)
    y_lower = y_test_pred + eps_lower
    y_upper = y_test_pred + eps_upper
    widths = y_upper - y_lower

    coverage = np.mean((y_test_orig >= y_lower) & (y_test_orig <= y_upper))
    return {
        'y_lower': y_lower,
        'y_upper': y_upper,
        'widths': widths,
        'coverage': coverage,
        'mean_width': np.mean(widths),
        'median_width': np.median(widths),
    }

# ------------------------------------------------------------------
# 计算两种置信水平下的区间
# ------------------------------------------------------------------
logger.info("Computing 95% intervals...")
res_95 = load_ngboost_and_intervals(MODEL_95_PATH, 0.95)

logger.info("Computing 99% intervals...")
res_99 = load_ngboost_and_intervals(MODEL_99_PATH, 0.99)

logger.info(f"95% -> Coverage: {res_95['coverage']:.2%}, Mean width: {res_95['mean_width']:.2f} kN")
logger.info(f"99% -> Coverage: {res_99['coverage']:.2%}, Mean width: {res_99['mean_width']:.2f} kN")

# ------------------------------------------------------------------
# 图 1：平均区间宽度柱状图 + 多级折线图（2个子图）
# ------------------------------------------------------------------
levels = [0.80, 0.90, 0.95, 0.99]
widths_95 = []
widths_99 = []
covs_95 = []
covs_99 = []

ng_95 = NGBoostTrainer(config, logger)
ng_95.load_model(MODEL_95_PATH)
ng_99 = NGBoostTrainer(config, logger)
ng_99.load_model(MODEL_99_PATH)

for lvl in levels:
    eps_l, eps_u = ng_95.predict_interval(X_test_with_var, confidence=lvl)
    widths_95.append(np.mean(y_test_pred + eps_u - (y_test_pred + eps_l)))
    covs_95.append(np.mean((y_test_orig >= y_test_pred + eps_l) & (y_test_orig <= y_test_pred + eps_u)))

    eps_l, eps_u = ng_99.predict_interval(X_test_with_var, confidence=lvl)
    widths_99.append(np.mean(y_test_pred + eps_u - (y_test_pred + eps_l)))
    covs_99.append(np.mean((y_test_orig >= y_test_pred + eps_l) & (y_test_orig <= y_test_pred + eps_u)))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 子图 1：平均宽度折线
ax = axes[0]
ax.plot([f"{l:.0%}" for l in levels], widths_95, marker='o', linewidth=2.5, label='95% CI model', color='#2E86AB')
ax.plot([f"{l:.0%}" for l in levels], widths_99, marker='s', linewidth=2.5, label='99% CI model', color='#F24236')
ax.set_ylabel('Mean Interval Width (kN)', fontsize=12)
ax.set_xlabel('Nominal Confidence Level', fontsize=12)
ax.set_title('Mean Prediction Interval Width vs Confidence Level', fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3)
for i, (w95, w99) in enumerate(zip(widths_95, widths_99)):
    ax.annotate(f'{w95:.0f}', (i, w95), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, color='#2E86AB')
    ax.annotate(f'{w99:.0f}', (i, w99), textcoords="offset points", xytext=(0, -15), ha='center', fontsize=9, color='#F24236')

# 子图 2：平均宽度柱状图
ax = axes[1]
x = np.arange(2)
width = 0.35
bars1 = ax.bar(x - width/2, [res_95['mean_width'], res_99['mean_width']], width, label='Mean Width', color=['#2E86AB', '#F24236'], edgecolor='black')
ax.set_ylabel('Mean Interval Width (kN)', fontsize=12)
ax.set_title('95% vs 99% Mean Interval Width Comparison', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(['95% CI', '99% CI'])
ax.set_ylim(0, max(res_99['mean_width'], res_95['mean_width']) * 1.2)
for bar in bars1:
    height = bar.get_height()
    ax.annotate(f'{height:.1f} kN',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
fig_path = OUTPUT_DIR / "ci_width_comparison_summary.png"
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
logger.info(f"Saved summary comparison to: {fig_path}")
plt.close()

# ------------------------------------------------------------------
# 图 2：测试集样本区间宽度分布（箱线图 + 小提琴图）
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

parts = ax.violinplot([res_95['widths'], res_99['widths']], positions=[1, 2], showmeans=False, showmedians=False, showextrema=False)
for pc, color in zip(parts['bodies'], ['#2E86AB', '#F24236']):
    pc.set_facecolor(color)
    pc.set_alpha(0.4)

bp = ax.boxplot([res_95['widths'], res_99['widths']], positions=[1, 2], widths=0.15, patch_artist=True,
                showmeans=True, meanline=True,
                medianprops=dict(color='black', linewidth=2),
                meanprops=dict(color='green', linewidth=2, linestyle='--'))
for patch, color in zip(bp['boxes'], ['#2E86AB', '#F24236']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_xticks([1, 2])
ax.set_xticklabels(['95% CI', '99% CI'])
ax.set_ylabel('Interval Width (kN)', fontsize=12)
ax.set_title('Distribution of Prediction Interval Widths on Test Set', fontsize=13)
ax.grid(True, axis='y', alpha=0.3)

# 添加统计标注
for pos, res, color in zip([1, 2], [res_95, res_99], ['#2E86AB', '#F24236']):
    ax.annotate(f"Mean: {res['mean_width']:.1f}\nMedian: {res['median_width']:.1f}",
                xy=(pos, res['mean_width']), xytext=(pos + 0.3, res['mean_width'] + 100),
                fontsize=10, color=color, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=color, lw=1))

plt.tight_layout()
fig_path = OUTPUT_DIR / "ci_width_distribution.png"
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
logger.info(f"Saved distribution comparison to: {fig_path}")
plt.close()

# ------------------------------------------------------------------
# 图 3：部分测试样本的区间条带对比图
# ------------------------------------------------------------------
n_samples_to_show = 15
indices = np.argsort(y_test_orig)[:n_samples_to_show]  # 取承载力最小的15个样本，便于展示

fig, ax = plt.subplots(figsize=(14, 7))

x_pos = np.arange(len(indices))
bar_height = 0.35

# 95% 区间
ax.barh(x_pos - bar_height/2, res_95['widths'][indices], height=bar_height, left=res_95['y_lower'][indices],
        color='#2E86AB', alpha=0.6, label='95% CI')
# 99% 区间
ax.barh(x_pos + bar_height/2, res_99['widths'][indices], height=bar_height, left=res_99['y_lower'][indices],
        color='#F24236', alpha=0.6, label='99% CI')

# 真实值和点预测
ax.scatter(y_test_orig[indices], x_pos, color='black', marker='D', s=60, zorder=5, label='Actual')
ax.scatter(y_test_pred[indices], x_pos, color='green', marker='x', s=60, zorder=5, label='XGBoost Point Pred')

ax.set_yticks(x_pos)
ax.set_yticklabels([f'Sample {i}' for i in indices])
ax.set_xlabel('Bearing Capacity (kN)', fontsize=12)
ax.set_title(f'Prediction Interval Comparison for {n_samples_to_show} Test Samples (Sorted by Actual)', fontsize=13)
ax.legend(loc='lower right')
ax.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
fig_path = OUTPUT_DIR / "ci_width_sample_bars.png"
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
logger.info(f"Saved sample bar comparison to: {fig_path}")
plt.close()

# ------------------------------------------------------------------
# 图 4：区间宽度 vs 实际值散点图（揭示异方差性）
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 7))

scatter1 = ax.scatter(y_test_orig, res_95['widths'], alpha=0.6, c='#2E86AB', edgecolors='black', s=50, label='95% CI Width')
scatter2 = ax.scatter(y_test_orig, res_99['widths'], alpha=0.6, c='#F24236', edgecolors='black', s=50, label='99% CI Width')

# 拟合趋势线
z95 = np.polyfit(y_test_orig, res_95['widths'], 1)
z99 = np.polyfit(y_test_orig, res_99['widths'], 1)
p95 = np.poly1d(z95)
p99 = np.poly1d(z99)
x_line = np.linspace(y_test_orig.min(), y_test_orig.max(), 100)
ax.plot(x_line, p95(x_line), '--', color='#2E86AB', linewidth=2)
ax.plot(x_line, p99(x_line), '--', color='#F24236', linewidth=2)

ax.set_xlabel('Actual Bearing Capacity (kN)', fontsize=12)
ax.set_ylabel('Interval Width (kN)', fontsize=12)
ax.set_title('Interval Width vs Actual Bearing Capacity (Heteroscedasticity)', fontsize=13)
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = OUTPUT_DIR / "ci_width_vs_actual.png"
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
logger.info(f"Saved width vs actual scatter to: {fig_path}")
plt.close()

logger.info("All comparison plots generated successfully!")
