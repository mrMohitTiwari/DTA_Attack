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
from sklearn.metrics import confusion_matrix
from config import RESULTS_DIR
from src.lr_model import LR_PARAMS


def plot_lr_dashboard(lr_orig, lr_def,
                      X_clean, X_adv, y_true,
                      metrics_orig, metrics_def):
    """
    6-panel dashboard for LR comparison.

    Panels:
    Row 1: F1 comparison | Attack success rate | Perturbation histogram
    Row 2: CM original   | CM defended         | Summary text
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    y_pred_orig = metrics_orig["y_pred"]
    y_pred_def  = metrics_def["y_pred"]

    cm_orig = confusion_matrix(y_true, y_pred_orig)
    cm_def  = confusion_matrix(y_true, y_pred_def)

    attack_idx   = np.where(y_true == 1)[0]
    orig_fooled = np.sum(y_pred_orig[attack_idx] == 0)
    def_fooled = np.sum(y_pred_def[attack_idx] == 0)

    f1_orig_clean = metrics_orig.get("f1_clean", 0)
    f1_orig_adv   = metrics_orig["f1"]
    f1_def_clean  = metrics_def.get("f1_clean", 0)
    f1_def_adv    = metrics_def["f1"]

    norms        = np.linalg.norm(X_adv - X_clean, axis=1)
    fig = plt.figure(figsize=(18, 11), constrained_layout=True)
    fig.suptitle(
        "Logistic Regression — Original vs Defended Model\n"
        "(FGSM Adversarial Attack + Adversarial Training Defence)",
        fontsize=15, fontweight='bold'
    )
    gs = gridspec.GridSpec(2, 3, figure=fig)

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
    vals  = [orig_fooled / n_atk * 100,
             def_fooled / n_atk * 100]
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

    # ── Panel 3: Perturbation distribution ────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    nz  = norms[norms > 1e-6]
    ax3.hist(nz, bins=50, color='#FF9800',
             edgecolor='white', alpha=0.85)
    ax3.axvline(nz.mean(), color='red', linestyle='--',
                label=f"mean={nz.mean():.4f}")
    ax3.set_xlabel("||x_adv - x||2")
    ax3.set_ylabel("Number of samples")
    ax3.set_title("FGSM Perturbation Size")
    ax3.legend(fontsize=9)

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

    # ── Panel 6: Summary text box ─────────────────────────────────
    ax6    = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    n_fooled_orig = np.sum(y_pred_orig[attack_idx] == 0)
    n_fooled_def  = np.sum(y_pred_def[attack_idx]  == 0)
    summary = (
        f"LOGISTIC REGRESSION SUMMARY\n"
        f"{'-'*34}\n\n"
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
        f"{(n_fooled_orig-n_fooled_def)/n_atk:.2%}"
    )
    ax6.text(0.03, 0.97, summary,
             transform=ax6.transAxes,
             fontsize=10, verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.6',
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


def plot_lr_fgsm_defense_summary(metrics_clean_orig, metrics_adv_orig,
                                 metrics_clean_def, metrics_adv_def,
                                 y_true_adv):
    """
    Focused chart for the user-facing LR FGSM result:
    F1 before/after adversarial training and confusion matrices.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    fig = plt.figure(figsize=(15, 8))
    fig.suptitle(
        "FGSM on Logistic Regression: Before vs After Adversarial Training",
        fontsize=15, fontweight="bold"
    )
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           height_ratios=[1.0, 1.15],
                           hspace=0.45, wspace=0.38)

    ax1 = fig.add_subplot(gs[0, :])
    labels = ["Original clean", "Original under FGSM",
              "Defended clean", "Defended under FGSM"]
    values = [metrics_clean_orig["f1"], metrics_adv_orig["f1"],
              metrics_clean_def["f1"], metrics_adv_def["f1"]]
    colors = ["#2F80ED", "#D64545", "#1F9D55", "#F2994A"]
    bars = ax1.bar(labels, values, color=colors,
                   edgecolor="white", width=0.58)
    for bar, value in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 value + 0.015, f"{value:.3f}",
                 ha="center", fontweight="bold")
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel("F1 score")
    ax1.set_title("F1 Score Drop from FGSM and Recovery After Defense")
    ax1.grid(axis="y", alpha=0.25)

    ax2 = fig.add_subplot(gs[1, 0])
    sns.heatmap(metrics_adv_orig["cm"], annot=True, fmt="d",
                cmap="Reds", cbar=False, ax=ax2,
                xticklabels=["Pred BENIGN", "Pred ATTACK"],
                yticklabels=["True BENIGN", "True ATTACK"])
    ax2.set_title("Original LR\nConfusion Matrix Under FGSM")

    ax3 = fig.add_subplot(gs[1, 1])
    sns.heatmap(metrics_adv_def["cm"], annot=True, fmt="d",
                cmap="Greens", cbar=False, ax=ax3,
                xticklabels=["Pred BENIGN", "Pred ATTACK"],
                yticklabels=["True BENIGN", "True ATTACK"])
    ax3.set_title("Defended LR\nConfusion Matrix Under FGSM")

    ax4 = fig.add_subplot(gs[1, 2])
    attack_idx = np.where(y_true_adv == 1)[0]
    orig_fooled = np.sum(metrics_adv_orig["y_pred"][attack_idx] == 0)
    def_fooled = np.sum(metrics_adv_def["y_pred"][attack_idx] == 0)
    n_attack = max(len(attack_idx), 1)
    vals = [orig_fooled / n_attack * 100,
            def_fooled / n_attack * 100]
    bars = ax4.bar(["Original", "Defended"], vals,
                   color=["#D64545", "#1F9D55"],
                   edgecolor="white", width=0.5)
    for bar, value in zip(bars, vals):
        ax4.text(bar.get_x() + bar.get_width() / 2,
                 value + 1, f"{value:.1f}%",
                 ha="center", fontweight="bold")
    ax4.set_ylim(0, 110)
    ax4.set_ylabel("Attack samples predicted BENIGN")
    ax4.set_title("FGSM Attack Success Rate")
    ax4.grid(axis="y", alpha=0.25)

    path = os.path.join(RESULTS_DIR, "lr_fgsm_defense_summary.png")
    plt.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
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
