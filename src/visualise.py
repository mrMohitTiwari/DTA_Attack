"""
visualise.py
Topic: matplotlib, seaborn, gridspec, confusion matrix,
       histogram, pie chart, bar chart, export CSV
What it does: all visualisation functions for DT pipeline
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy  as np
import pandas as pd
import matplotlib.pyplot   as plt
import matplotlib.gridspec as gridspec
import seaborn             as sns
from sklearn.metrics import confusion_matrix
from config import RESULTS_DIR


def plot_dashboard(results, X_clean, X_adv, y_true):
    """
    6-panel dashboard showing DTA attack impact on original model.

    Panels:
      1. F1 before vs after DTA
      2. Attack success pie chart
      3. Confusion matrix after DTA
      4. Perturbation size distribution
      5. DTA sparsity (features changed per sample)
      6. Top 10 most targeted features
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    f1_clean     = results["f1_clean"]
    f1_adv       = results["f1_adv"]
    y_pred_clean = results["y_pred_clean"]
    y_pred_adv   = results["y_pred_adv"]
    n_flipped    = results["n_flipped"]
    attack_idx   = np.where(y_true == 1)[0]
    not_flipped  = len(attack_idx) - n_flipped
    norms        = np.linalg.norm(X_adv - X_clean, axis=1)
    feat_changed = np.sum(np.abs(X_adv - X_clean) > 1e-6, axis=1)

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        "DTA (Decision Tree Attack) — Full Analysis",
        fontsize=16, fontweight='bold', y=0.98
    )
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.45, wspace=0.38)

    # Panel 1: F1 bar chart
    ax1  = fig.add_subplot(gs[0, 0])
    bars = ax1.bar(
        ['Clean', 'After DTA'],
        [f1_clean, f1_adv],
        color=['#2196F3', '#F44336'],
        width=0.5, edgecolor='white'
    )
    for bar, val in zip(bars, [f1_clean, f1_adv]):
        ax1.text(
            bar.get_x()+bar.get_width()/2, val+0.01,
            f"{val:.4f}", ha='center',
            fontsize=11, fontweight='bold'
        )
    ax1.set_ylim(0, 1.15)
    ax1.set_ylabel("F1 Score")
    ax1.set_title("F1: Before vs After DTA")
    ax1.axhline(0.5, color='gray', linestyle='--',
                linewidth=0.8, label='0.5 baseline')
    ax1.legend(fontsize=9)

    # Panel 2: Success pie
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.pie(
        [n_flipped, not_flipped],
        labels=[f'Fooled\n({n_flipped})',
                f'Not fooled\n({not_flipped})'],
        colors=['#F44336', '#2196F3'],
        autopct='%1.1f%%', startangle=90,
        textprops={'fontsize': 10}
    )
    ax2.set_title("DTA Success Rate")

    # Panel 3: Confusion matrix
    ax3 = fig.add_subplot(gs[0, 2])
    cm  = confusion_matrix(y_true, y_pred_adv)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Reds',
        xticklabels=['Pred BENIGN', 'Pred ATTACK'],
        yticklabels=['True BENIGN', 'True ATTACK'],
        ax=ax3, cbar=False
    )
    ax3.set_title("Confusion Matrix\n(After DTA)")
    ax3.set_ylabel("True label")
    ax3.set_xlabel("Predicted")

    # Panel 4: Perturbation histogram
    ax4 = fig.add_subplot(gs[1, 0])
    nz  = norms[norms > 1e-6]
    ax4.hist(nz, bins=50, color='#FF9800',
             edgecolor='white', alpha=0.85)
    ax4.axvline(nz.mean(), color='red', linestyle='--',
                label=f"mean={nz.mean():.4f}")
    ax4.set_xlabel("||x_adv - x||₂")
    ax4.set_ylabel("Samples")
    ax4.set_title("Perturbation Size Distribution")
    ax4.legend(fontsize=9)

    # Panel 5: Features changed per sample
    ax5 = fig.add_subplot(gs[1, 1])
    uniq, cnts = np.unique(feat_changed, return_counts=True)
    ax5.bar(uniq.astype(str), cnts,
            color='#9C27B0', edgecolor='white', alpha=0.85)
    ax5.set_xlabel("Features changed per sample")
    ax5.set_ylabel("Number of samples")
    ax5.set_title("DTA Sparsity")

    # Panel 6: Most targeted features
    ax6 = fig.add_subplot(gs[1, 2])
    freq    = np.sum(np.abs(X_adv - X_clean) > 1e-6, axis=0)
    top_idx = np.argsort(freq)[::-1][:10]
    ax6.barh(
        [f"f{i}" for i in top_idx[::-1]],
        freq[top_idx[::-1]],
        color='#009688', edgecolor='white', alpha=0.85
    )
    ax6.set_xlabel("Times targeted by DTA")
    ax6.set_title("Top 10 Most Targeted Features")

    path = os.path.join(RESULTS_DIR, "dta_analysis.png")
    plt.savefig(path, dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"Saved: {path}")


def plot_comparison_dashboard(comparison, X_clean,
                               X_adv, y_true):
    """
    6-panel dashboard comparing original vs defended DT model.

    Panels:
      1. F1 grouped bar — clean and under attack
      2. Attack success rate before vs after defence
      3. Confusion matrix — original model under DTA
      4. Confusion matrix — defended model under DTA
      5. Side-by-side F1 horizontal bars
      6. Summary text box
    """
    def plot_multi_seed_dashboard(summary):
   

     os.makedirs(RESULTS_DIR, exist_ok=True)

    per_run = summary["per_run"]

    seeds = [r["seed"] for r in per_run]

    f1_clean = [r["f1_clean_orig"] for r in per_run]
    f1_orig  = [r["f1_adv_orig"] for r in per_run]
    f1_def   = [r["f1_adv_def"] for r in per_run]

    fooled_orig = [
        100 * r["n_fooled_orig"] / r["n_total"]
        for r in per_run
    ]

    fooled_def = [
        100 * r["n_fooled_def"] / r["n_total"]
        for r in per_run
    ]

    recovery = [
        d - o for d, o in zip(f1_def, f1_orig)
    ]

    # ============================================================
    # FIGURE
    # ============================================================

    fig = plt.figure(figsize=(18, 12))

    fig.suptitle(
        "Multi-Seed Statistical Evaluation — DTA Robustness",
        fontsize=18,
        fontweight='bold',
        y=0.98
    )

    gs = gridspec.GridSpec(
        2, 3,
        figure=fig,
        hspace=0.42,
        wspace=0.35
    )

    # ============================================================
    # PANEL 1 — F1 across seeds
    # ============================================================

    ax1 = fig.add_subplot(gs[0, 0])

    ax1.plot(
        seeds, f1_clean,
        marker='o',
        linewidth=2.5,
        label='Clean F1'
    )

    ax1.plot(
        seeds, f1_orig,
        marker='s',
        linewidth=2.5,
        label='Original under DTA'
    )

    ax1.plot(
        seeds, f1_def,
        marker='^',
        linewidth=2.5,
        label='Defended under DTA'
    )

    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel("Random Seed")
    ax1.set_ylabel("F1 Score")
    ax1.set_title("F1 Across Random Seeds")

    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9)

    # ============================================================
    # PANEL 2 — Mean ± std
    # ============================================================

    ax2 = fig.add_subplot(gs[0, 1])

    means = [
        summary["mean_f1_clean"],
        summary["mean_f1_orig_adv"],
        summary["mean_f1_def_adv"]
    ]

    stds = [
        summary["std_f1_clean"],
        summary["std_f1_orig_adv"],
        summary["std_f1_def_adv"]
    ]

    labels = [
        "Clean",
        "Original\nUnder DTA",
        "Defended\nUnder DTA"
    ]

    bars = ax2.bar(
        labels,
        means,
        yerr=stds,
        capsize=8,
        color=['#2196F3', '#F44336', '#4CAF50'],
        edgecolor='white'
    )

    for bar, val in zip(bars, means):
        ax2.text(
            bar.get_x() + bar.get_width()/2,
            val + 0.02,
            f"{val:.3f}",
            ha='center',
            fontsize=10,
            fontweight='bold'
        )

    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("Mean F1")
    ax2.set_title("Mean ± Std F1")

    # ============================================================
    # PANEL 3 — Attack success rates
    # ============================================================

    ax3 = fig.add_subplot(gs[0, 2])

    x = np.arange(len(seeds))
    width = 0.35

    bars1 = ax3.bar(
        x - width/2,
        fooled_orig,
        width,
        label='Original',
        color='#F44336',
        edgecolor='white'
    )

    bars2 = ax3.bar(
        x + width/2,
        fooled_def,
        width,
        label='Defended',
        color='#4CAF50',
        edgecolor='white'
    )

    ax3.set_xticks(x)
    ax3.set_xticklabels(seeds)

    ax3.set_ylim(0, 105)

    ax3.set_xlabel("Random Seed")
    ax3.set_ylabel("Attack Success Rate (%)")

    ax3.set_title("Attack Success Rate")

    ax3.legend(fontsize=9)

    # ============================================================
    # PANEL 4 — F1 recovery
    # ============================================================

    ax4 = fig.add_subplot(gs[1, 0])

    bars = ax4.bar(
        seeds,
        recovery,
        color='#9C27B0',
        edgecolor='white',
        alpha=0.9
    )

    for bar, val in zip(bars, recovery):
        ax4.text(
            bar.get_x() + bar.get_width()/2,
            val + 0.01,
            f"{val:.3f}",
            ha='center',
            fontsize=9
        )

    ax4.set_xlabel("Random Seed")
    ax4.set_ylabel("F1 Recovery")

    ax4.set_title(
        "Recovery After Adversarial Training"
    )

    ax4.grid(axis='y', alpha=0.3)

    # ============================================================
    # PANEL 5 — Boxplot stability
    # ============================================================

    ax5 = fig.add_subplot(gs[1, 1])

    sns.boxplot(
        data=[f1_clean, f1_orig, f1_def],
        palette=['#2196F3', '#F44336', '#4CAF50'],
        ax=ax5
    )

    ax5.set_xticklabels([
        'Clean',
        'Original\nDTA',
        'Defended\nDTA'
    ])

    ax5.set_ylabel("F1 Score")

    ax5.set_title("Variance / Stability Analysis")

    # ============================================================
    # PANEL 6 — Statistical summary
    # ============================================================

    ax6 = fig.add_subplot(gs[1, 2])

    ax6.axis('off')

    summary_txt = (
        f"SUMMARY\n"
        f"{'─'*42}\n\n"
        f"Runs evaluated:      {len(seeds)}\n\n"

        f"Clean F1\n"
        f"  Mean:              "
        f"{summary['mean_f1_clean']:.4f}\n"
        f"  Std:               "
        f"{summary['std_f1_clean']:.4f}\n\n"

        f"Original model under DTA\n"
        f"  Mean F1:           "
        f"{summary['mean_f1_orig_adv']:.4f}\n"
        f"  Std:               "
        f"{summary['std_f1_orig_adv']:.4f}\n\n"

        f"Defended model under DTA\n"
        f"  Mean F1:           "
        f"{summary['mean_f1_def_adv']:.4f}\n"
        f"  Std:               "
        f"{summary['std_f1_def_adv']:.4f}\n\n"

        f"Average F1 recovery:\n"
        f"  {summary['mean_f1_def_adv'] - summary['mean_f1_orig_adv']:+.4f}"
    )

    ax6.text(
        0.05,
        0.95,
        summary_txt,
        transform=ax6.transAxes,
        fontsize=11,
        verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(
            boxstyle='round',
            facecolor='#f5f5f5',
            edgecolor='#cccccc',
            alpha=0.9
        )
    )

    # ============================================================
    # SAVE
    # ============================================================

    path = os.path.join(
        RESULTS_DIR,
        "multi_seed_dashboard.png"
    )

    plt.savefig(
        path,
        dpi=150,
        bbox_inches='tight',
        facecolor='white'
    )

    plt.show()

    print(f"Saved: {path}")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    f1_oc        = comparison["f1_orig_clean"]
    f1_oa        = comparison["f1_orig_adv"]
    f1_dc        = comparison["f1_def_clean"]
    f1_da        = comparison["f1_def_adv"]
    y_orig_adv   = comparison["y_orig_adv"]
    y_def_adv    = comparison["y_def_adv"]
    n_attack     = comparison["n_attack"]
    orig_flipped = comparison["orig_flipped"]
    def_flipped  = comparison["def_flipped"]

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        "Adversarial Training Defence — Original vs Defended Model",
        fontsize=16, fontweight='bold', y=0.98
    )
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.45, wspace=0.40)

    # Panel 1: F1 grouped bar
    ax1   = fig.add_subplot(gs[0, 0])
    x     = np.arange(2)
    width = 0.35
    bars1 = ax1.bar(
        x - width/2,
        [f1_oc, f1_oa], width,
        label='Original model',
        color='#F44336', edgecolor='white'
    )
    bars2 = ax1.bar(
        x + width/2,
        [f1_dc, f1_da], width,
        label='Defended model',
        color='#4CAF50', edgecolor='white'
    )
    for bars in [bars1, bars2]:
        for bar in bars:
            ax1.text(
                bar.get_x()+bar.get_width()/2,
                bar.get_height()+0.01,
                f"{bar.get_height():.3f}",
                ha='center', fontsize=9,
                fontweight='bold'
            )
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Clean data', 'Under DTA attack'])
    ax1.set_ylim(0, 1.2)
    ax1.set_ylabel("F1 Score")
    ax1.set_title("F1: Original vs Defended")
    ax1.legend(fontsize=9)
    ax1.axhline(0.5, color='gray', linestyle='--',
                linewidth=0.8, alpha=0.6)

    # Panel 2: Attack success rate
    ax2  = fig.add_subplot(gs[0, 1])
    cats = ['Original\n(no defence)', 'Defended\n(adv. training)']
    vals = [
        orig_flipped / n_attack * 100,
        def_flipped  / n_attack * 100
    ]
    bars = ax2.bar(cats, vals,
                   color=['#F44336', '#4CAF50'],
                   edgecolor='white', width=0.5)
    for bar, val in zip(bars, vals):
        ax2.text(
            bar.get_x()+bar.get_width()/2,
            val+0.5, f"{val:.1f}%",
            ha='center', fontsize=11,
            fontweight='bold'
        )
    ax2.set_ylabel("Attack success rate (%)")
    ax2.set_title("DTA Success Rate\nBefore vs After Defence")
    ax2.set_ylim(0, 110)

    # Panel 3: CM original
    ax3     = fig.add_subplot(gs[0, 2])
    cm_orig = confusion_matrix(y_true, y_orig_adv)
    sns.heatmap(
        cm_orig, annot=True, fmt='d', cmap='Reds',
        xticklabels=['Pred BEN', 'Pred ATK'],
        yticklabels=['True BEN', 'True ATK'],
        ax=ax3, cbar=False
    )
    ax3.set_title("Confusion Matrix\nOriginal model under DTA")

    # Panel 4: CM defended
    ax4    = fig.add_subplot(gs[1, 0])
    cm_def = confusion_matrix(y_true, y_def_adv)
    sns.heatmap(
        cm_def, annot=True, fmt='d', cmap='Greens',
        xticklabels=['Pred BEN', 'Pred ATK'],
        yticklabels=['True BEN', 'True ATK'],
        ax=ax4, cbar=False
    )
    ax4.set_title("Confusion Matrix\nDefended model under DTA")

    # Panel 5: Side-by-side F1 bars
    ax5       = fig.add_subplot(gs[1, 1])
    metrics   = ['F1 (clean)', 'F1 (DTA attack)']
    orig_vals = [f1_oc, f1_oa]
    def_vals  = [f1_dc, f1_da]
    y_pos     = np.array([0.7, 0.3])
    ax5.barh(y_pos+0.12, orig_vals, height=0.18,
             color='#F44336', label='Original', alpha=0.85)
    ax5.barh(y_pos-0.06, def_vals,  height=0.18,
             color='#4CAF50', label='Defended', alpha=0.85)
    ax5.set_yticks(y_pos)
    ax5.set_yticklabels(metrics)
    ax5.set_xlim(0, 1.1)
    ax5.set_xlabel("F1 Score")
    ax5.set_title("Side-by-side F1 Comparison")
    ax5.legend(fontsize=9)
    for i, (ov, dv) in enumerate(zip(orig_vals, def_vals)):
        ax5.text(ov+0.01, y_pos[i]+0.12,
                 f"{ov:.3f}", va='center', fontsize=9)
        ax5.text(dv+0.01, y_pos[i]-0.06,
                 f"{dv:.3f}", va='center', fontsize=9)

    # Panel 6: Summary text
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    summary = (
        f"SUMMARY\n"
        f"{'─'*38}\n\n"
        f"Original model\n"
        f"  F1 (clean data):    {f1_oc:.4f}\n"
        f"  F1 (under DTA):     {f1_oa:.4f}\n"
        f"  Attacks fooled:     {orig_flipped}/{n_attack}\n"
        f"  Success rate:       {orig_flipped/n_attack:.2%}\n\n"
        f"Defended model\n"
        f"  F1 (clean data):    {f1_dc:.4f}\n"
        f"  F1 (under DTA):     {f1_da:.4f}\n"
        f"  Attacks fooled:     {def_flipped}/{n_attack}\n"
        f"  Success rate:       {def_flipped/n_attack:.2%}\n\n"
        f"F1 recovery:          {f1_da - f1_oa:+.4f}\n"
        f"Attack reduction:     "
        f"{(orig_flipped-def_flipped)/n_attack:.2%}"
    )
    ax6.text(
        0.05, 0.95, summary,
        transform=ax6.transAxes,
        fontsize=11, verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round',
                  facecolor='#f5f5f5',
                  alpha=0.8,
                  edgecolor='#cccccc')
    )

    path = os.path.join(
        RESULTS_DIR,
        "adversarial_training_comparison.png"
    )
    plt.savefig(path, dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"Saved: {path}")


def export_csv(X_clean, X_adv, y_true,
               y_pred_clean, y_pred_adv):
    """
    Export sample-level results to CSV.
    Shows original vs adversarial prediction per row.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    n     = min(200, len(X_clean))
    norms = np.linalg.norm(X_adv - X_clean, axis=1)
    fc    = np.sum(np.abs(X_adv - X_clean) > 1e-6, axis=1)

    rows = [{
        "sample_index":       i,
        "true_label":         int(y_true[i]),
        "pred_clean":         int(y_pred_clean[i]),
        "pred_adversarial":   int(y_pred_adv[i]),
        "attack_succeeded":   int(
            y_true[i] == 1 and
            y_pred_clean[i] == 1 and
            y_pred_adv[i] == 0
        ),
        "n_features_changed": int(fc[i]),
        "perturbation_l2":    round(float(norms[i]), 6),
    } for i in range(n)]

    path = os.path.join(RESULTS_DIR, "dta_results.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Saved: {path}")


if __name__ == "__main__":
    print("visualise.py loaded — functions available:")
    print("  plot_dashboard(results, X_clean, X_adv, y_true)")
    print("  plot_comparison_dashboard(comparison, X_clean, X_adv, y_true)")
    print("  export_csv(X_clean, X_adv, y_true, y_pred_clean, y_pred_adv)")
