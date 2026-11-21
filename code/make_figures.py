from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'results'
OUT = ROOT / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

# Journal-style defaults: compact sans serif typography,
# minimal ink, colour-blind-safe model distinction and no decorative background.
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'axes.linewidth': 0.7,
    'xtick.major.width': 0.7,
    'ytick.major.width': 0.7,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'ps.fonttype': 42,
    'figure.dpi': 180,
})

C_LOG = '#4D4D4D'
C_LGB = '#1F77B4'
C_LIGHT = '#A6A6A6'


def clean_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out')


def panel(ax, letter):
    ax.text(-0.12, 1.04, letter, transform=ax.transAxes, fontsize=9,
            fontweight='bold', va='bottom', ha='left')


def save(fig, n):
    # Explicit margins avoid tight_layout warnings with nested GridSpec panels
    # and preserve the external annotations used in Figures 1 and 2.
    if n == 1:
        fig.subplots_adjust(left=0.12, right=0.985, bottom=0.20, top=0.96,
                            wspace=0.42, hspace=0.66)
    elif n == 2:
        fig.subplots_adjust(left=0.13, right=0.985, bottom=0.31, top=0.96,
                            wspace=0.48, hspace=0.45)
    elif n == 4:
        fig.subplots_adjust(left=0.14, right=0.985, bottom=0.13, top=0.97,
                            wspace=0.34, hspace=0.48)
    fig.savefig(OUT / f'Fig{n}.eps', bbox_inches='tight', facecolor='white')
    plt.close(fig)


# Fig. 1: pooled validation performance plus experiment-level model-gap uncertainty.
v = pd.read_csv(SRC / 'validation_performance.csv')
c = pd.read_csv(SRC / 'combined_purge_performance.csv')
v = pd.concat([v, c], ignore_index=True)
gap = pd.read_csv(SRC / 'validation_gap_uncertainty.csv')
fig = plt.figure(figsize=(7.25, 4.15))
gs = fig.add_gridspec(2, 2, height_ratios=[1.55, 1.0], hspace=0.62, wspace=0.42)
axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
sets = [
    ('a', axs[0],
     ['Row random', 'Learner grouped', 'Learner + class grouped', 'Learner + teacher grouped'],
     ['Row random', 'Learner grouped', 'Learner + class', 'Learner + teacher']),
    ('b', axs[1],
     ['Standard LOEO', 'Training-size-matched LOEO', 'Learner-purged LOEO', 'Learner + class + teacher-purged LOEO'],
     ['Standard LOEO', 'Size-matched', 'Learner-purged', 'Joint purge'])
]
for letter, ax, keys, labels in sets:
    y = np.arange(len(keys))[::-1]
    logv = np.array([v[(v.validation == k) & (v.model == 'Logistic regression')].iloc[0].auroc for k in keys])
    lgbv = np.array([v[(v.validation == k) & (v.model == 'LightGBM')].iloc[0].auroc for k in keys])
    for yy, a, b in zip(y, logv, lgbv):
        ax.plot([a, b], [yy, yy], color=C_LIGHT, lw=0.8, zorder=1)
    ax.scatter(logv, y, s=24, marker='o', color=C_LOG, label='SGD logistic regression', zorder=3)
    ax.scatter(lgbv, y, s=25, marker='s', color=C_LGB, label='LightGBM', zorder=3)
    ax.set_yticks(y, labels)
    ax.tick_params(axis='y', labelsize=7.2, pad=2)
    ax.set_xlim(0.695, 0.765)
    ax.set_xlabel('Pooled AUROC')
    clean_ax(ax); panel(ax, letter)
axs[1].legend(frameon=False, loc='lower right', handletextpad=0.5, borderaxespad=0.2)
ax = fig.add_subplot(gs[1, :])
order = ['Row random', 'Learner grouped', 'Learner + class grouped', 'Learner + teacher grouped',
         'Standard LOEO', 'Training-size-matched LOEO', 'Learner-purged LOEO',
         'Learner + class + teacher-purged LOEO']
labels = ['Row-random', 'Learner', 'Learner + class', 'Learner + teacher',
          'Standard LOEO', 'Size-matched', 'Learner-purged', 'Joint purge']
gap = gap.set_index('validation').loc[order].reset_index()
x = np.arange(len(gap))
med = gap.median_delta_auroc.to_numpy()
lo = gap.bootstrap_95_low.to_numpy(); hi = gap.bootstrap_95_high.to_numpy()
ax.errorbar(x, med, yerr=[med-lo, hi-med], fmt='o', color=C_LGB, ecolor=C_LGB,
            capsize=2.5, lw=1.0, ms=4, zorder=3)
ax.axhline(0, color=C_LIGHT, lw=0.8, ls=':')
ax.set_xticks(x, labels, rotation=15, ha='right')
ax.set_ylabel('Median paired AUROC gap\n(LightGBM − SGD logistic regression)')
ax.set_ylim(min(-0.002, lo.min()-0.001), hi.max()+0.003)
clean_ax(ax); panel(ax, 'c')
save(fig, 1)


# Fig. 2: predictor-set estimates and paired experiment-level uncertainty.
auc = pd.read_csv(SRC / 'feature_auc_summary.csv')
abl = pd.read_csv(SRC / 'feature_ablation_uncertainty.csv')
fig, axs = plt.subplots(1, 2, figsize=(7.25, 3.15), gridspec_kw={'wspace': 0.42})
ax = axs[0]
sel = auc[(auc.validation == 'Learner-purged LOEO') &
          (auc.model.isin(['Logistic regression (SGD)', 'Logistic regression (LBFGS)']))].copy()
plot_order = [
    ('Logistic regression (SGD)', 'Full transportable', 'SGD: full set'),
    ('Logistic regression (SGD)', 'Complete learner history', 'SGD: learner history'),
    ('Logistic regression (LBFGS)', 'Full transportable', 'L-BFGS: full set'),
]
for y, (model, feature_set, label) in enumerate(plot_order[::-1]):
    r = sel[(sel.model == model) & (sel.feature_set == feature_set)].iloc[0]
    colour = C_LOG if 'SGD' in model else '#888888'
    ax.errorbar(r.median_auroc, y,
                xerr=[[r.median_auroc-r.bootstrap_95_low], [r.bootstrap_95_high-r.median_auroc]],
                fmt='o', color=colour, ecolor=colour, capsize=2.3, ms=4)
ax.set_yticks(np.arange(len(plot_order)), [x[2] for x in plot_order[::-1]])
ax.set_xlim(0.715, 0.765)
ax.set_xlabel('Median experiment-level AUROC (95% bootstrap CI)')
sgd_delta = abl[abl.comparison == 'Logistic SGD: learner history minus full set'].iloc[0]
ax.text(0.02, -0.24,
        f"Paired Δ history − full (SGD): {sgd_delta.median_difference:+.3f} "
        f"[{sgd_delta.median_bootstrap_95_low:+.3f}, {sgd_delta.median_bootstrap_95_high:+.3f}]",
        transform=ax.transAxes, fontsize=7.0, va='top')
clean_ax(ax); panel(ax, 'a')

ax = axs[1]
ctx_labels = [
    ('LightGBM: context removal under row-random', 'Row random'),
    ('LightGBM: context removal under learner-purged LOEO', 'Learner-purged LOEO'),
]
for y, (key, label) in enumerate(ctx_labels[::-1]):
    r = abl[abl.comparison == key].iloc[0]
    ax.errorbar(r.median_difference, y,
                xerr=[[r.median_difference-r.median_bootstrap_95_low],
                      [r.median_bootstrap_95_high-r.median_difference]],
                fmt='s', color=C_LGB, ecolor=C_LGB, capsize=2.3, ms=4)
ax.axvline(0, color=C_LIGHT, lw=0.8, ls=':')
ax.set_yticks([0, 1], ['Learner-purged LOEO', 'Row random'])
ax.set_xlabel('AUROC change after removing context\n(No context − full set; 95% bootstrap CI)')
interaction = abl[abl.comparison.str.startswith('LightGBM: row-random context effect minus')].iloc[0]
ax.text(0.02, -0.28,
        f"Row-random − learner-purged context effect: {interaction.median_difference:+.3f} "
        f"[{interaction.median_bootstrap_95_low:+.3f}, {interaction.median_bootstrap_95_high:+.3f}]",
        transform=ax.transAxes, fontsize=7.0, va='top')
clean_ax(ax); panel(ax, 'b')
save(fig, 2)


# Fig. 3: experiment-level calibration and local probability updating.
cal = pd.read_csv(SRC / 'calibration_summary.csv')
recal = pd.read_csv(SRC / 'local_recalibration_comparison.csv')
fig, axs = plt.subplots(1, 2, figsize=(7.25, 3.75), gridspec_kw={'wspace': 0.42})
ax = axs[0]
calibration_styles = [
    ('Logistic regression', C_LOG, 'o'),
    ('Logistic regression (LBFGS)', '#888888', 'D'),
    ('LightGBM', C_LGB, 's'),
]
for mi, (model, colour, marker) in enumerate(calibration_styles):
    sub = cal[(cal.scope == 'Experiment-level') & (cal.model == model)]
    if sub.empty:
        continue
    for j, metric in enumerate(['Calibration intercept', 'Calibration slope']):
        r = sub[sub.metric == metric].iloc[0]
        xpos = j + (mi - 1) * 0.16
        ax.errorbar(xpos, r.value, yerr=[[r.value-r.q1], [r.q3-r.value]],
                    fmt=marker, color=colour, markerfacecolor=colour,
                    ms=4.0, lw=0.9, capsize=2.3,
                    label=('SGD logistic regression' if model == 'Logistic regression' else 'L-BFGS logistic regression' if model == 'Logistic regression (LBFGS)' else model) if j == 0 else None)
ax.axhline(0, color=C_LIGHT, lw=0.8, ls=':', xmin=0.03, xmax=0.48)
ax.axhline(1, color=C_LIGHT, lw=0.8, ls=':', xmin=0.52, xmax=0.97)
ax.set_xticks([0, 1], ['Intercept', 'Slope'])
ax.set_ylabel('Experiment-level estimate (median and IQR)')
ax.legend(frameon=False, loc='upper left', fontsize=6.6)
clean_ax(ax); panel(ax, 'a')

ax = axs[1]
positions = {
    'No local update': 0,
    'Calibration-cohort prevalence': 1,
    'Intercept update': 2,
    'Intercept + slope update': 3,
}
recal_styles = [
    ('Logistic regression', C_LOG, 'o', -0.18),
    ('Logistic regression (LBFGS)', '#888888', 'D', -0.06),
    ('Prevalence-only reference', '#777777', '^', 0.06),
    ('LightGBM', C_LGB, 's', 0.18),
]
for model, colour, marker, offset in recal_styles:
    sub = recal[recal.model == model]
    for version, xpos in positions.items():
        r = sub[sub.probability_version == version]
        if r.empty:
            continue
        r = r.iloc[0]
        mean = r.mean_experiment_brier
        lo = r.mean_brier_bootstrap_95_low
        hi = r.mean_brier_bootstrap_95_high
        ax.errorbar(xpos + offset, mean, yerr=[[mean-lo], [hi-mean]],
                    fmt=marker, color=colour, ecolor=colour, ms=3.9,
                    lw=0.85, capsize=2.1,
                    label=('SGD logistic regression' if model == 'Logistic regression' else 'L-BFGS logistic regression' if model == 'Logistic regression (LBFGS)' else model) if version == 'No local update' or model == 'Prevalence-only reference' else None)
ax.set_xticks([0, 1, 2, 3], ['No update', 'Local prevalence', 'Intercept', 'Intercept + slope'])
ax.set_ylabel('Mean experiment-level Brier score')
ax.tick_params(axis='x', labelrotation=12, labelsize=7)
ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.42),
          fontsize=6.2, ncol=2, columnspacing=1.2, handletextpad=0.5)
clean_ax(ax); panel(ax, 'b')
fig.subplots_adjust(left=0.11, right=0.985, bottom=0.38, top=0.94, wspace=0.42)
fig.savefig(OUT / 'Fig3.eps', bbox_inches='tight', facecolor='white')
plt.close(fig)


# Fig. 4: capacity metrics with experiment-cluster bootstrap intervals and paired differences.
cap = pd.read_csv(SRC / 'capacity_model_uncertainty.csv')
diff = pd.read_csv(SRC / 'capacity_difference_uncertainty.csv')
fig, axs = plt.subplots(2, 2, figsize=(6.85, 4.35), sharex='col',
                        gridspec_kw={'height_ratios': [1.2, 1.0], 'hspace': 0.48, 'wspace': 0.28})
model_styles = [
    ('Logistic regression (SGD)', C_LOG, 'o', '-'),
    ('Logistic regression (LBFGS)', '#888888', 'D', '--'),
    ('LightGBM', C_LGB, 's', '-'),
]
for ax, metric, low_col, high_col, ylabel, letter in [
    (axs[0, 0], 'precision', 'precision_bootstrap_95_low', 'precision_bootstrap_95_high', 'Precision', 'a'),
    (axs[0, 1], 'recall', 'recall_bootstrap_95_low', 'recall_bootstrap_95_high', 'Recall', 'b'),
]:
    for model, colour, marker, linestyle in model_styles:
        sub = cap[cap.model == model].sort_values('capacity')
        if sub.empty:
            continue
        x = sub.capacity.to_numpy() * 100
        y = sub[metric].to_numpy()
        lo = sub[low_col].to_numpy(); hi = sub[high_col].to_numpy()
        display_model = {
            'Logistic regression (SGD)': 'SGD logistic regression',
            'Logistic regression (LBFGS)': 'L-BFGS logistic regression',
            'LightGBM': 'LightGBM',
        }[model]
        ax.errorbar(x, y, yerr=[y-lo, hi-y], color=colour, marker=marker,
                    linestyle=linestyle, lw=1.0, ms=3.5, capsize=2.0,
                    label=display_model)
    ax.set_ylabel(ylabel)
    # Shared x-axes show numeric tick labels only on the lower row.
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    clean_ax(ax); panel(ax, letter)
axs[0, 0].legend(frameon=False, loc='best', fontsize=6.4)
axs[0, 1].legend(frameon=False, loc='best', fontsize=6.4)

for ax, metric, ylabel, letter in [
    (axs[1, 0], 'precision', 'Δ Precision (LightGBM − logistic)', 'c'),
    (axs[1, 1], 'recall', 'Δ Recall (LightGBM − logistic)', 'd'),
]:
    for comparison, colour, marker, linestyle in [
        ('LightGBM minus Logistic regression (SGD)', C_LGB, 'o', '-'),
        ('LightGBM minus Logistic regression (LBFGS)', '#888888', 'D', '--'),
    ]:
        sub = diff[(diff.comparison == comparison) & (diff.metric == metric)].sort_values('capacity')
        if sub.empty:
            continue
        x = sub.capacity.to_numpy() * 100
        y = sub.difference.to_numpy()
        lo = sub.bootstrap_95_low.to_numpy(); hi = sub.bootstrap_95_high.to_numpy()
        ax.errorbar(x, y, yerr=[y-lo, hi-y], color=colour, marker=marker,
                    linestyle=linestyle, lw=1.0, ms=3.5, capsize=2.0, label=('LightGBM − SGD logistic regression' if '(SGD)' in comparison else 'LightGBM − L-BFGS logistic regression'))
    ax.axhline(0, color=C_LIGHT, lw=0.8, ls=':')
    ax.set_ylabel('Δ Precision' if metric == 'precision' else 'Δ Recall')
    ax.set_xlabel('Review capacity within experiment (%)')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    clean_ax(ax); panel(ax, letter)
axs[1, 0].legend(frameon=False, loc='upper left', fontsize=6.1)
axs[1, 1].legend(frameon=False, loc='upper left', fontsize=6.1)
save(fig, 4)


# Supplementary figures used in ESM1.
sens = pd.read_csv(SRC / 'sensitivity_summary.csv')
hlgb = pd.read_csv(SRC / 'horizon_lightgbm.csv')
# Merge primary model horizon values already frozen in result tables.
log_rows = sens[(sens.model == 'Logistic regression') & sens.analysis.isin(['7-day outcome','14-day outcome','30-day outcome','Archive closure outcome'])]
lgb_map = {'7-day outcome': None, '30-day outcome': None, 'Archive closure outcome': None}
for _, r in hlgb.iterrows(): lgb_map[r.analysis] = r.auroc
# primary 14-day LGB under experiment-blocked learner-purged CV is in feature/sensitivity source table
lgb14 = pd.read_csv(SRC / 'feature_blocks.csv')
lgb14 = lgb14[(lgb14.validation == 'Learner-purged experiment-grouped CV') & (lgb14.model == 'LightGBM') & (lgb14.feature_set == 'Full transportable')].iloc[0].auroc
lgb_map['14-day outcome'] = lgb14
order = ['7-day outcome','14-day outcome','30-day outcome','Archive closure outcome']
lab = ['7 days','14 days','30 days','Archive closure']
fig, ax = plt.subplots(figsize=(3.3, 2.35))
x = np.arange(4)
logvals = [log_rows[log_rows.analysis == a].iloc[0].auroc for a in order]
lgbvals = [lgb_map[a] for a in order]
ax.plot(x, logvals, marker='o', color=C_LOG, lw=1.1, ms=4, label='SGD logistic regression')
ax.plot(x, lgbvals, marker='s', color=C_LGB, lw=1.1, ms=4, label='LightGBM')
ax.set_xticks(x, lab); ax.set_ylabel('AUROC'); ax.set_ylim(0.68, 0.73)
ax.legend(frameon=False, loc='lower right'); clean_ax(ax); fig.tight_layout(pad=0.7)
fig.savefig(OUT/'FigS1.eps',bbox_inches='tight',facecolor='white')
plt.close(fig)

# repeated-learner sensitivity
rb = sens[sens.analysis.isin(['Learner-balanced evaluation','Earliest record per learner'])]
fig, ax = plt.subplots(figsize=(3.3, 2.35))
y=np.array([1,0]); labels=['Learner-balanced','Earliest record per learner']
for yy,a in zip(y,['Learner-balanced evaluation','Earliest record per learner']):
    xlog=rb[(rb.analysis==a)&(rb.model=='Logistic regression')].iloc[0].auroc
    xlgb=rb[(rb.analysis==a)&(rb.model=='LightGBM')].iloc[0].auroc
    ax.plot([xlog,xlgb],[yy,yy],color=C_LIGHT,lw=.9)
    ax.scatter(xlog,yy,s=24,color=C_LOG,marker='o',label='SGD logistic regression' if yy==1 else None)
    ax.scatter(xlgb,yy,s=25,color=C_LGB,marker='s',label='LightGBM' if yy==1 else None)
ax.set_yticks(y,labels); ax.set_xlabel('AUROC'); ax.set_xlim(.665,.710); ax.set_xticks([.67,.68,.69,.70])
ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.27),
          ncol=2, columnspacing=1.4, handlelength=1.3); clean_ax(ax); fig.tight_layout(pad=.8)
fig.savefig(OUT/'FigS2.eps',bbox_inches='tight',facecolor='white')
plt.close(fig)

print('Figures written to', OUT)
