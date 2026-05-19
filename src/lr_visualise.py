"""
lr_visualise.py
Topic: matplotlib subplots, seaborn heatmap, grouped bar charts,
       weight visualisation, ROC curve, precision-recall tradeoff
What it does: creates a comprehensive 9-panel dashboard comparing
              original vs defended LR across all metrics
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy  as np
import pandas as pd
import matplotlib.pyplot     as plt
import matplotlib.gridspec   as gridspec
import matplotlib.patches    as mpatches
import seaborn               as sns
from sklearn.metrics import (confusion_matrix, roc_curve,
                              auc, precision_recall_curve)
from config import RESULTS_DIR


def plot_lr_dashboard(lr_orig, lr_def,
                      X_clean, X_adv, y_true,
                      metrics_orig, metrics_def):
    """
    9-panel comprehensive dashboard for LR comparison.

    Panels:
    Row 1: F1 comparison | Attack success rate | Weight comparison
    Row 2: CM original   | CM defended         | F1/Precision/Recall bars
    Row 3: ROC curve     | Perturbation dist   | Summary text
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    y_pred_orig = metrics_orig["y_pred"]
    y_pred_def  = metrics_def["y_pred"]

    cm_orig = confusion_matrix(y_true, y_pred_orig)
    cm_def  = confusion_matrix(y_true, y_pred_def)

    attack_idx   = np.where(y_true == 1)[0]
    orig_fooled  = np.sum(
        (metrics_orig["y_pred"][attack_idx] == 1) &
        # was originally correct, now wrong after adv
        # y_pred_orig here is prediction on ADV data
        (y_pred_orig[attack_idx] == 0)
    )
    def_fooled = np.sum(y_pred_def[attack_idx] == 0)

    f1_orig_clean = metrics_orig.get("f1_clean", 0)
    f1_orig_adv   = metrics_orig["f1"]
    f1_def_clean  = metrics_def.get("f1_clean", 0)
    f1_def_adv    = metrics_def["f1"]

    norms        = np.linalg.norm(X_adv - X_clean, axis=1)
    feat_changed = np.sum(
        np.abs(X_adv - X_clean) > 1e-6, axis=1
    )

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(
        "Logistic Regression — Original vs Defended Model\n"
        "(FGSM Adversarial Attack + Adversarial Training Defence)",
        fontsize=15, fontweight='bold', y=0.98
    )
    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.50, wspace=0.40)

    # ── Panel 1: F1 grouped bar ────────────────────────────────────
    ax1  = fig.add_subplot(gs[0, 0])
    x    = np.arange(2)
    w    = 0.35
    b1   = ax1.bar(x - w/2,
                   [f1_orig_clean, f1_orig_adv],
                   w, label='Original LR',
                   color='#F44336', edgecolor='white',
                   alpha=0.9)
    b2   = ax1.bar(x + w/2,
                   [f1_def_clean, f1_def_adv],
                   w, label='Defended LR',
                   color='#4CAF50', edgecolor='white',
                   alpha=0.9)
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x()+bar.get_width()/2,
                     h+0.01, f"{h:.3f}",
                     ha='center', fontsize=9,
                     fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Clean data', 'Under FGSM'])
    ax1.set_ylim(0, 1.2)
    ax1.set_ylabel("F1 Score")
    ax1.set_title("F1: Original vs Defended LR")
    ax1.legend(fontsize=9)
    ax1.axhline(0.5, color='gray', linestyle='--',
                linewidth=0.8, alpha=0.5)

    # ── Panel 2: Attack success rate ───────────────────────────────
    ax2  = fig.add_subplot(gs[0, 1])
    n_atk = len(attack_idx)
    cats  = ['Original\n(no defence)', 'Defended\n(adv. training)']
    vals  = [def_fooled / n_atk * 100,
             np.sum(y_pred_def[attack_idx] == 0) / n_atk * 100]
    cols  = ['#F44336', '#4CAF50']
    bars  = ax2.bar(cats, vals, color=cols,
                    edgecolor='white', width=0.5, alpha=0.9)
    for bar, val in zip(bars, vals):
        ax2.text(bar.get_x()+bar.get_width()/2,
                 val+0.5, f"{val:.1f}%",
                 ha='center', fontsize=11,
                 fontweight='bold')
    ax2.set_ylabel("FGSM attack success rate (%)")
    ax2.set_title("FGSM Success Rate\nBefore vs After Defence")
    ax2.set_ylim(0, 115)

    # ── Panel 3: Weight comparison bar ────────────────────────────
    ax3    = fig.add_subplot(gs[0, 2])
    w_orig = lr_orig.coef_[0]
    w_def  = lr_def.coef_[0]
    w_diff = np.abs(w_def - w_orig)
    top10  = np.argsort(w_diff)[::-1][:10]
    y_pos  = np.arange(10)
    ax3.barh(y_pos,      w_orig[top10[::-1]],
             height=0.4, left=0,
             color='#F44336', alpha=0.8,
             label='Original')
    ax3.barh(y_pos+0.4,  w_def[top10[::-1]],
             height=0.4,
             color='#4CAF50', alpha=0.8,
             label='Defended')
    ax3.set_yticks(y_pos+0.2)
    ax3.set_yticklabels(
        [f"f{i}" for i in top10[::-1]],
        fontsize=9
    )
    ax3.set_xlabel("Weight value")
    ax3.set_title("Top 10 Weight Changes\n"
                  "from Adversarial Training")
    ax3.legend(fontsize=9)
    ax3.axvline(0, color='gray', linewidth=0.8)

    # ── Panel 4: Confusion matrix — original ──────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    sns.heatmap(cm_orig, annot=True, fmt='d', cmap='Reds',
                xticklabels=['Pred BEN','Pred ATK'],
                yticklabels=['True BEN','True ATK'],
                ax=ax4, cbar=False)
    ax4.set_title("Confusion Matrix\nOriginal LR under FGSM")

    # ── Panel 5: Confusion matrix — defended ──────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    sns.heatmap(cm_def, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Pred BEN','Pred ATK'],
                yticklabels=['True BEN','True ATK'],
                ax=ax5, cbar=False)
    ax5.set_title("Confusion Matrix\nDefended LR under FGSM")

    # ── Panel 6: Precision / Recall / F1 triple bar ───────────────
    ax6    = fig.add_subplot(gs[1, 2])
    labels = ['Precision', 'Recall', 'F1']
    orig_v = [metrics_orig['precision'],
              metrics_orig['recall'],
              metrics_orig['f1']]
    def_v  = [metrics_def['precision'],
              metrics_def['recall'],
              metrics_def['f1']]
    x6     = np.arange(3)
    ax6.bar(x6 - 0.2, orig_v, 0.35,
            label='Original', color='#F44336',
            edgecolor='white', alpha=0.9)
    ax6.bar(x6 + 0.2, def_v,  0.35,
            label='Defended', color='#4CAF50',
            edgecolor='white', alpha=0.9)
    for i, (ov, dv) in enumerate(zip(orig_v, def_v)):
        ax6.text(i-0.2, ov+0.01, f"{ov:.2f}",
                 ha='center', fontsize=8)
        ax6.text(i+0.2, dv+0.01, f"{dv:.2f}",
                 ha='center', fontsize=8)
    ax6.set_xticks(x6)
    ax6.set_xticklabels(labels)
    ax6.set_ylim(0, 1.2)
    ax6.set_ylabel("Score")
    ax6.set_title("Precision / Recall / F1\nUnder FGSM Attack")
    ax6.legend(fontsize=9)

    # ── Panel 7: ROC curve both models ────────────────────────────
    ax7 = fig.add_subplot(gs[2, 0])
    for model, label, color in [
        (lr_orig, 'Original LR', '#F44336'),
        (lr_def,  'Defended LR', '#4CAF50')
    ]:
        proba = model.predict_proba(X_adv)[:, 1]
        fpr, tpr, _ = roc_curve(y_true, proba)
        roc_auc     = auc(fpr, tpr)
        ax7.plot(fpr, tpr, color=color, linewidth=2,
                 label=f"{label} (AUC={roc_auc:.3f})")
    ax7.plot([0,1],[0,1],'k--', linewidth=0.8, alpha=0.5)
    ax7.set_xlabel("False Positive Rate")
    ax7.set_ylabel("True Positive Rate")
    ax7.set_title("ROC Curve\nUnder FGSM Attack")
    ax7.legend(fontsize=9)
    ax7.set_xlim(0, 1)
    ax7.set_ylim(0, 1.05)

    # ── Panel 8: Perturbation distribution ────────────────────────
    ax8 = fig.add_subplot(gs[2, 1])
    nz  = norms[norms > 1e-6]
    ax8.hist(nz, bins=50, color='#FF9800',
             edgecolor='white', alpha=0.85)
    ax8.axvline(nz.mean(), color='red', linestyle='--',
                label=f"mean={nz.mean():.4f}")
    ax8.set_xlabel("||x_adv - x||₂")
    ax8.set_ylabel("Number of samples")
    ax8.set_title("FGSM Perturbation Size\n"
                  "(all features perturbed — unlike DTA)")
    ax8.legend(fontsize=9)

    # ── Panel 9: Summary text box ──────────────────────────────────
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    n_fooled_orig = np.sum(y_pred_orig[attack_idx] == 0)
    n_fooled_def  = np.sum(y_pred_def[attack_idx]  == 0)
    summary = (
        f"LOGISTIC REGRESSION SUMMARY\n"
        f"{'─'*36}\n\n"
        f"Model: LR  C={LR_PARAMS['C']:.0f}\n"
        f"Attack: FGSM  eps=0.1\n\n"
        f"Original model\n"
        f"  F1 (clean):      {f1_orig_clean:.4f}\n"
        f"  F1 (FGSM):       {f1_orig_adv:.4f}\n"
        f"  Attacks fooled:  {n_fooled_orig}/{n_atk}\n"
        f"  Success rate:    "
        f"{n_fooled_orig/n_atk:.2%}\n\n"
        f"Defended model\n"
        f"  F1 (clean):      {f1_def_clean:.4f}\n"
        f"  F1 (FGSM):       {f1_def_adv:.4f}\n"
        f"  Attacks fooled:  {n_fooled_def}/{n_atk}\n"
        f"  Success rate:    "
        f"{n_fooled_def/n_atk:.2%}\n\n"
        f"Recovery:          "
        f"{f1_def_adv - f1_orig_adv:+.4f}\n"
        f"Attack reduction:  "
        f"{(n_fooled_orig-n_fooled_def)/n_atk:.2%}\n\n"
        f"Paper finding:\n"
        f"  LR most robust against adv\n"
        f"  attacks — small gradient\n"
        f"  means weak perturbations"
    )
    ax9.text(0.03, 0.97, summary,
             transform=ax9.transAxes,
             fontsize=10, verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round',
                       facecolor='#f5f5f5',
                       alpha=0.8,
                       edgecolor='#cccccc'))

    path = os.path.join(
        RESULTS_DIR,
        "lr_adversarial_comparison.png"
    )
    plt.savefig(path, dpi=150,
                bbox_inches='tight',
                facecolor='white')
    plt.show()
    print(f"Saved: {path}")
    return path


def plot_lr_vs_dt_comparison(lr_results, dt_results):
    """
    Side by side comparison of LR and DT results.
    This directly connects to the paper's finding that
    LR is more robust than DT against adversarial attacks.

    Args:
        lr_results : dict with lr f1 scores
        dt_results : dict with dt f1 scores
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "LR vs DT — Adversarial Robustness Comparison\n"
        "(Reproducing ELAT Paper Finding: "
        "LR most robust classifier)",
        fontsize=13, fontweight='bold'
    )

    # ── Plot 1: F1 clean comparison ───────────────────────────────
    categories = ['Clean F1', 'Under Attack F1', 'Defended F1']
    lr_vals    = [
        lr_results['f1_clean'],
        lr_results['f1_adv'],
        lr_results['f1_defended']
    ]
    dt_vals    = [
        dt_results['f1_clean'],
        dt_results['f1_adv'],
        dt_results['f1_defended']
    ]
    x = np.arange(3)
    axes[0].bar(x-0.2, lr_vals, 0.35,
                label='Logistic Regression',
                color='#2196F3', edgecolor='white',
                alpha=0.9)
    axes[0].bar(x+0.2, dt_vals, 0.35,
                label='Decision Tree',
                color='#FF9800', edgecolor='white',
                alpha=0.9)
    for i, (lv, dv) in enumerate(zip(lr_vals, dt_vals)):
        axes[0].text(i-0.2, lv+0.01, f"{lv:.2f}",
                     ha='center', fontsize=8)
        axes[0].text(i+0.2, dv+0.01, f"{dv:.2f}",
                     ha='center', fontsize=8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories, fontsize=9)
    axes[0].set_ylim(0, 1.2)
    axes[0].set_ylabel("F1 Score")
    axes[0].set_title("F1 Scores: LR vs DT")
    axes[0].legend(fontsize=9)

    # ── Plot 2: Attack success rate ───────────────────────────────
    models   = ['LR (FGSM)', 'DT (DTA)']
    orig_asr = [lr_results['orig_asr'], dt_results['orig_asr']]
    def_asr  = [lr_results['def_asr'],  dt_results['def_asr']]
    x2       = np.arange(2)
    axes[1].bar(x2-0.2, orig_asr, 0.35,
                label='No defence',
                color='#F44336', edgecolor='white',
                alpha=0.9)
    axes[1].bar(x2+0.2, def_asr, 0.35,
                label='With adv. training',
                color='#4CAF50', edgecolor='white',
                alpha=0.9)
    for i, (ov, dv) in enumerate(zip(orig_asr, def_asr)):
        axes[1].text(i-0.2, ov+0.5, f"{ov:.1f}%",
                     ha='center', fontsize=9,
                     fontweight='bold')
        axes[1].text(i+0.2, dv+0.5, f"{dv:.1f}%",
                     ha='center', fontsize=9,
                     fontweight='bold')
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(models)
    axes[1].set_ylim(0, 120)
    axes[1].set_ylabel("Attack success rate (%)")
    axes[1].set_title("Attack Success Rate\nBefore vs After Defence")
    axes[1].legend(fontsize=9)

    # ── Plot 3: F1 drop and recovery ─────────────────────────────
    models3  = ['LR', 'DT']
    drop     = [
        lr_results['f1_clean'] - lr_results['f1_adv'],
        dt_results['f1_clean'] - dt_results['f1_adv']
    ]
    recovery = [
        lr_results['f1_defended'] - lr_results['f1_adv'],
        dt_results['f1_defended'] - dt_results['f1_adv']
    ]
    x3 = np.arange(2)
    axes[2].bar(x3-0.2, drop,     0.35,
                label='F1 drop (attack damage)',
                color='#F44336', edgecolor='white',
                alpha=0.9)
    axes[2].bar(x3+0.2, recovery, 0.35,
                label='F1 recovery (defence)',
                color='#4CAF50', edgecolor='white',
                alpha=0.9)
    for i, (dr, rc) in enumerate(zip(drop, recovery)):
        axes[2].text(i-0.2, dr+0.01, f"{dr:.3f}",
                     ha='center', fontsize=9)
        axes[2].text(i+0.2, rc+0.01, f"{rc:.3f}",
                     ha='center', fontsize=9)
    axes[2].set_xticks(x3)
    axes[2].set_xticklabels(models3)
    axes[2].set_ylim(0, 1.15)
    axes[2].set_ylabel("F1 points")
    axes[2].set_title("Attack Damage vs Defence Recovery\n"
                      "(higher recovery = better defence)")
    axes[2].legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "lr_vs_dt_comparison.png")
    plt.savefig(path, dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"Saved: {path}")