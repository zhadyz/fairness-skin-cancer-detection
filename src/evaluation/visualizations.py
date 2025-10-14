"""
Visualization Pipeline for Fairness Evaluation

Generates comprehensive visualizations for fairness analysis:
- ROC curves per Fitzpatrick skin type
- Calibration plots (reliability diagrams)
- Confusion matrices per FST
- Fairness gap bar charts
- Performance comparison plots

Author: HOLLOWED_EYES
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from sklearn.metrics import (
    roc_curve,
    confusion_matrix,
    precision_recall_curve,
    auc
)


# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


class FairnessVisualizer:
    """
    Comprehensive fairness visualization toolkit.

    Generates publication-ready plots for fairness analysis of skin cancer
    detection models across Fitzpatrick skin types.

    Example:
        >>> visualizer = FairnessVisualizer(save_dir='experiments/baseline/figures')
        >>> visualizer.plot_roc_curves_per_fst(y_true, y_prob, fst_labels)
        >>> visualizer.plot_fairness_gaps(auroc_per_fst)
    """

    def __init__(self, save_dir: str = 'figures', dpi: int = 300):
        """
        Initialize visualizer.

        Args:
            save_dir: Directory to save figures
            dpi: Resolution for saved figures
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

        # FST color palette (light to dark skin tones)
        self.fst_colors = {
            1: '#FDE6C4',  # Very light
            2: '#F5D3A1',  # Light
            3: '#D4A574',  # Light-medium
            4: '#B87E4E',  # Medium
            5: '#8B5A3C',  # Dark
            6: '#5C3A21'   # Very dark
        }

    def plot_roc_curves_per_fst(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        fst_labels: np.ndarray,
        target_class: int = 0,
        save_name: str = 'roc_curves_per_fst.png'
    ):
        """
        Plot ROC curves for each Fitzpatrick skin type (overlaid).

        Args:
            y_true: True labels (N,)
            y_prob: Predicted probabilities (N, num_classes)
            fst_labels: FST labels (N,)
            target_class: Target class for binary ROC
            save_name: Filename to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Convert to binary for target class
        y_true_binary = (y_true == target_class).astype(int)
        y_score = y_prob[:, target_class]

        fst_groups = sorted(np.unique(fst_labels))

        for fst in fst_groups:
            if fst not in [1, 2, 3, 4, 5, 6]:
                continue

            fst_mask = fst_labels == fst

            if not np.any(fst_mask):
                continue

            fst_y_true = y_true_binary[fst_mask]
            fst_y_score = y_score[fst_mask]

            # Compute ROC curve
            fpr, tpr, _ = roc_curve(fst_y_true, fst_y_score)
            roc_auc = auc(fpr, tpr)

            # Plot
            color = self.fst_colors.get(fst, 'black')
            ax.plot(
                fpr, tpr,
                color=color,
                lw=2.5,
                label=f'FST-{fst} (AUC = {roc_auc:.3f})',
                alpha=0.8
            )

        # Diagonal reference line
        ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Chance')

        ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
        ax.set_title(
            'ROC Curves by Fitzpatrick Skin Type',
            fontsize=16,
            fontweight='bold',
            pad=20
        )
        ax.legend(loc='lower right', fontsize=11, frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        print(f"Saved ROC curves to {self.save_dir / save_name}")

    def plot_calibration_curves_per_fst(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        fst_labels: np.ndarray,
        num_bins: int = 10,
        save_name: str = 'calibration_curves_per_fst.png'
    ):
        """
        Plot calibration (reliability) curves for each FST.

        Args:
            y_true: True labels (N,)
            y_prob: Predicted probabilities (N, num_classes)
            fst_labels: FST labels (N,)
            num_bins: Number of calibration bins
            save_name: Filename to save figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Get predicted class and confidence
        y_pred = np.argmax(y_prob, axis=1)
        confidences = np.max(y_prob, axis=1)
        accuracies = (y_pred == y_true).astype(float)

        fst_groups = sorted(np.unique(fst_labels))

        for fst in fst_groups:
            if fst not in [1, 2, 3, 4, 5, 6]:
                continue

            fst_mask = fst_labels == fst

            if not np.any(fst_mask):
                continue

            fst_confidences = confidences[fst_mask]
            fst_accuracies = accuracies[fst_mask]

            # Compute calibration curve
            bin_edges = np.linspace(0, 1, num_bins + 1)
            bin_centers = []
            bin_accuracies = []

            for i in range(num_bins):
                bin_mask = (fst_confidences >= bin_edges[i]) & (fst_confidences < bin_edges[i + 1])

                if np.sum(bin_mask) > 0:
                    bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
                    bin_accuracies.append(np.mean(fst_accuracies[bin_mask]))

            # Plot
            color = self.fst_colors.get(fst, 'black')
            ax.plot(
                bin_centers, bin_accuracies,
                'o-',
                color=color,
                lw=2.5,
                markersize=8,
                label=f'FST-{fst}',
                alpha=0.8
            )

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.5, label='Perfect Calibration')

        ax.set_xlabel('Confidence', fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
        ax.set_title(
            'Calibration Curves by Fitzpatrick Skin Type',
            fontsize=16,
            fontweight='bold',
            pad=20
        )
        ax.legend(loc='upper left', fontsize=11, frameon=True, shadow=True)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        print(f"Saved calibration curves to {self.save_dir / save_name}")

    def plot_confusion_matrices_per_fst(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        fst_labels: np.ndarray,
        class_names: Optional[List[str]] = None,
        save_name: str = 'confusion_matrices_per_fst.png'
    ):
        """
        Plot confusion matrices for each FST in a grid.

        Args:
            y_true: True labels (N,)
            y_pred: Predicted labels (N,)
            fst_labels: FST labels (N,)
            class_names: Names of disease classes
            save_name: Filename to save figure
        """
        fst_groups = [1, 2, 3, 4, 5, 6]
        num_classes = len(np.unique(y_true))

        if class_names is None:
            class_names = [f'Class {i}' for i in range(num_classes)]

        # Create 2x3 grid for FST I-VI
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for idx, fst in enumerate(fst_groups):
            fst_mask = fst_labels == fst

            if not np.any(fst_mask):
                axes[idx].axis('off')
                continue

            fst_y_true = y_true[fst_mask]
            fst_y_pred = y_pred[fst_mask]

            # Compute confusion matrix
            cm = confusion_matrix(fst_y_true, fst_y_pred)
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            # Plot
            sns.heatmap(
                cm_normalized,
                annot=True,
                fmt='.2f',
                cmap='Blues',
                ax=axes[idx],
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Normalized Count'}
            )

            axes[idx].set_title(f'FST-{fst} (n={np.sum(fst_mask)})', fontsize=14, fontweight='bold')
            axes[idx].set_xlabel('Predicted', fontsize=12)
            axes[idx].set_ylabel('True', fontsize=12)

        plt.suptitle(
            'Confusion Matrices by Fitzpatrick Skin Type',
            fontsize=18,
            fontweight='bold',
            y=0.995
        )
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        print(f"Saved confusion matrices to {self.save_dir / save_name}")

    def plot_fairness_gaps(
        self,
        auroc_per_fst: Dict[int, float],
        save_name: str = 'fairness_gaps.png'
    ):
        """
        Plot AUROC comparison across FST groups with gap visualization.

        Args:
            auroc_per_fst: Dictionary mapping FST to AUROC
            save_name: Filename to save figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Bar chart of AUROC per FST
        fst_groups = sorted(auroc_per_fst.keys())
        aurocs = [auroc_per_fst[fst] for fst in fst_groups]
        colors = [self.fst_colors.get(fst, 'gray') for fst in fst_groups]

        bars = ax1.bar(
            range(len(fst_groups)),
            aurocs,
            color=colors,
            edgecolor='black',
            linewidth=1.5,
            alpha=0.8
        )

        # Add value labels on bars
        for bar, auroc in zip(bars, aurocs):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f'{auroc:.3f}',
                ha='center',
                va='bottom',
                fontsize=11,
                fontweight='bold'
            )

        ax1.set_xlabel('Fitzpatrick Skin Type', fontsize=14, fontweight='bold')
        ax1.set_ylabel('AUROC', fontsize=14, fontweight='bold')
        ax1.set_title('AUROC by Fitzpatrick Skin Type', fontsize=16, fontweight='bold')
        ax1.set_xticks(range(len(fst_groups)))
        ax1.set_xticklabels([f'FST-{fst}' for fst in fst_groups])
        ax1.set_ylim([0, 1.0])
        ax1.grid(True, axis='y', alpha=0.3)

        # Add horizontal line for overall mean
        mean_auroc = np.mean(aurocs)
        ax1.axhline(mean_auroc, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_auroc:.3f}')
        ax1.legend(fontsize=11)

        # Plot 2: Light vs Dark comparison
        light_fst = [1, 2, 3]
        dark_fst = [4, 5, 6]

        light_aurocs = [auroc_per_fst[fst] for fst in light_fst if fst in auroc_per_fst]
        dark_aurocs = [auroc_per_fst[fst] for fst in dark_fst if fst in auroc_per_fst]

        light_mean = np.mean(light_aurocs) if light_aurocs else 0
        dark_mean = np.mean(dark_aurocs) if dark_aurocs else 0
        gap = light_mean - dark_mean

        comparison_data = [light_mean, dark_mean]
        comparison_labels = ['Light Skin\n(FST I-III)', 'Dark Skin\n(FST IV-VI)']
        comparison_colors = ['#F5D3A1', '#5C3A21']

        bars2 = ax2.bar(
            range(2),
            comparison_data,
            color=comparison_colors,
            edgecolor='black',
            linewidth=1.5,
            alpha=0.8,
            width=0.6
        )

        # Add value labels
        for bar, value in zip(bars2, comparison_data):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f'{value:.3f}',
                ha='center',
                va='bottom',
                fontsize=12,
                fontweight='bold'
            )

        # Add gap annotation
        ax2.annotate(
            f'Gap: {gap:.3f}',
            xy=(0.5, max(comparison_data)),
            xytext=(0.5, max(comparison_data) + 0.1),
            ha='center',
            fontsize=14,
            fontweight='bold',
            color='red',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7)
        )

        ax2.set_ylabel('Average AUROC', fontsize=14, fontweight='bold')
        ax2.set_title('Light vs Dark Skin Performance Gap', fontsize=16, fontweight='bold')
        ax2.set_xticks(range(2))
        ax2.set_xticklabels(comparison_labels, fontsize=12)
        ax2.set_ylim([0, 1.0])
        ax2.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        print(f"Saved fairness gaps to {self.save_dir / save_name}")

    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        save_name: str = 'training_history.png'
    ):
        """
        Plot training history (loss, accuracy, AUROC).

        Args:
            history: Dictionary with 'train_loss', 'val_loss', etc.
            save_name: Filename to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        epochs = range(1, len(history['train_loss']) + 1)

        # Plot 1: Loss
        axes[0, 0].plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train Loss')
        axes[0, 0].plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Val Loss')
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Accuracy
        axes[0, 1].plot(epochs, history['train_acc'], 'b-', linewidth=2, label='Train Accuracy')
        axes[0, 1].plot(epochs, history['val_acc'], 'r-', linewidth=2, label='Val Accuracy')
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Accuracy', fontsize=12)
        axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: AUROC
        axes[1, 0].plot(epochs, history['val_auroc'], 'g-', linewidth=2, label='Val AUROC')
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('AUROC', fontsize=12)
        axes[1, 0].set_title('Validation AUROC', fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Learning Rate
        axes[1, 1].plot(epochs, history['learning_rate'], 'orange', linewidth=2, label='Learning Rate')
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
        axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle('Training History', fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        print(f"Saved training history to {self.save_dir / save_name}")


if __name__ == "__main__":
    # Test visualizations
    print("Testing Fairness Visualizations...")

    np.random.seed(42)

    # Generate dummy data
    n_samples = 1000
    num_classes = 7

    y_true = np.random.randint(0, num_classes, n_samples)
    y_pred = np.random.randint(0, num_classes, n_samples)
    y_prob = np.random.rand(n_samples, num_classes)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    fst_labels = np.random.choice([1, 2, 3, 4, 5, 6], n_samples)

    # Dummy AUROC data
    auroc_per_fst = {1: 0.92, 2: 0.90, 3: 0.88, 4: 0.82, 5: 0.78, 6: 0.75}

    # Create visualizer
    visualizer = FairnessVisualizer(save_dir='test_figures')

    # Test visualizations
    visualizer.plot_roc_curves_per_fst(y_true, y_prob, fst_labels, target_class=0)
    visualizer.plot_calibration_curves_per_fst(y_true, y_prob, fst_labels)
    visualizer.plot_confusion_matrices_per_fst(y_true, y_pred, fst_labels)
    visualizer.plot_fairness_gaps(auroc_per_fst)

    # Test training history
    history = {
        'train_loss': np.linspace(1.5, 0.3, 50).tolist(),
        'val_loss': np.linspace(1.6, 0.4, 50).tolist(),
        'train_acc': np.linspace(0.4, 0.92, 50).tolist(),
        'val_acc': np.linspace(0.35, 0.88, 50).tolist(),
        'val_auroc': np.linspace(0.5, 0.85, 50).tolist(),
        'learning_rate': (np.logspace(-4, -6, 50)).tolist()
    }
    visualizer.plot_training_history(history)

    print("\nVisualization tests PASSED!")
    print(f"Figures saved to test_figures/")
