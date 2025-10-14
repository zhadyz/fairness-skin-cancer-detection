"""
Comprehensive Fairness Evaluation Script

Loads trained baseline models and performs comprehensive fairness evaluation
across Fitzpatrick skin types with detailed reporting and visualizations.

Usage:
    python experiments/baseline/evaluate_fairness.py \
        --checkpoint experiments/baseline/checkpoints/resnet50_best.pth \
        --model resnet50 \
        --config configs/baseline_config.yaml

Author: HOLLOWED_EYES
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import argparse
import yaml
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from datetime import datetime

from src.models.resnet_baseline import create_resnet50_model
from src.models.efficientnet_baseline import create_efficientnet_model
from src.models.inception_baseline import create_inception_model
from src.evaluation.fairness_metrics import FairnessMetrics
from src.evaluation.visualizations import FairnessVisualizer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(model_name: str, num_classes: int):
    """Create model based on name."""
    if model_name == 'resnet50':
        return create_resnet50_model(num_classes=num_classes, pretrained=False)
    elif model_name.startswith('efficientnet'):
        variant = model_name.split('_')[-1] if '_' in model_name else 'b4'
        return create_efficientnet_model(variant=variant, num_classes=num_classes, pretrained=False)
    elif model_name == 'inception_v3':
        return create_inception_model(num_classes=num_classes, pretrained=False)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def create_dummy_test_loader(config: dict):
    """Create dummy test loader (replace with actual dataset)."""
    print("\n" + "="*80)
    print("WARNING: Using dummy test data")
    print("Replace with actual test dataset loader")
    print("="*80 + "\n")

    num_test = 500
    img_size = config['data']['img_size']
    num_classes = config['model']['num_classes']

    test_images = torch.randn(num_test, 3, img_size, img_size)
    test_labels = torch.randint(0, num_classes, (num_test,))
    test_fst = torch.randint(1, 7, (num_test,))

    test_dataset = TensorDataset(test_images, test_labels, test_fst)

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0
    )

    return test_loader


def evaluate_model(model, test_loader, device, num_classes):
    """Evaluate model and compute fairness metrics."""
    model.eval()
    model.to(device)

    all_y_true = []
    all_y_pred = []
    all_y_prob = []
    all_fst = []

    print("Running inference on test set...")

    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                images, labels, fst_labels = batch
            else:
                images, labels = batch
                fst_labels = torch.ones(len(labels))

            images = images.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_y_true.extend(labels.cpu().numpy())
            all_y_pred.extend(preds.cpu().numpy())
            all_y_prob.extend(probs.cpu().numpy())
            all_fst.extend(fst_labels.cpu().numpy())

    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)
    y_prob = np.array(all_y_prob)
    fst_labels = np.array(all_fst)

    print(f"Evaluated {len(y_true)} samples")

    # Compute fairness metrics
    print("\nComputing fairness metrics...")
    fairness_metrics = FairnessMetrics(num_classes=num_classes)
    results = fairness_metrics.evaluate(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        fst_labels=fst_labels,
        target_class=0  # Melanoma class
    )

    return {
        'fairness_results': results,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'fst_labels': fst_labels
    }


def generate_fairness_report(
    model_name: str,
    fairness_results,
    save_path: Path
):
    """Generate comprehensive fairness report in Markdown format."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    report_lines = [
        f"# Fairness Evaluation Report: {model_name}",
        f"\n**Generated:** {timestamp}",
        f"\n## Overall Performance",
        f"\n- **Overall AUROC:** {fairness_results.overall_auroc:.4f}",
        f"- **Overall Accuracy:** {fairness_results.overall_accuracy:.4f}",
        f"\n## Performance by Fitzpatrick Skin Type",
        f"\n| FST | AUROC | Accuracy | Sensitivity | Specificity | ECE | Samples |",
        "| --- | ----- | -------- | ----------- | ----------- | --- | ------- |"
    ]

    for fst in sorted(fairness_results.auroc_per_fst.keys()):
        auroc = fairness_results.auroc_per_fst[fst]
        acc = fairness_results.accuracy_per_fst[fst]
        sens = fairness_results.sensitivity_per_fst[fst]
        spec = fairness_results.specificity_per_fst[fst]
        ece = fairness_results.ece_per_fst[fst]
        samples = fairness_results.samples_per_fst[fst]

        report_lines.append(
            f"| {fst} | {auroc:.4f} | {acc:.4f} | {sens:.4f} | {spec:.4f} | {ece:.4f} | {samples} |"
        )

    report_lines.extend([
        f"\n## Fairness Gaps",
        f"\n### AUROC Gaps",
        f"- **Max-Min Gap:** {fairness_results.auroc_gap:.4f}",
        f"- **Light-Dark Gap (FST I-III vs IV-VI):** {fairness_results.auroc_gap_light_dark:.4f}",
        f"\n### Equalized Odds",
        f"- **Equal Opportunity Difference:** {fairness_results.equal_opportunity_diff:.4f}",
        f"  - Maximum difference in True Positive Rate across groups",
        f"\n### Demographic Parity",
        f"- **Demographic Parity Difference:** {fairness_results.demographic_parity_diff:.4f}",
        f"  - Maximum difference in positive prediction rate across groups",
        f"\n## Light vs Dark Skin Performance",
        f"\n### Light Skin (FST I-III)"
    ])

    light_fst = [1, 2, 3]
    light_aurocs = [fairness_results.auroc_per_fst[fst] for fst in light_fst if fst in fairness_results.auroc_per_fst]
    if light_aurocs:
        report_lines.extend([
            f"- Average AUROC: {np.mean(light_aurocs):.4f}",
            f"- Min AUROC: {np.min(light_aurocs):.4f}",
            f"- Max AUROC: {np.max(light_aurocs):.4f}"
        ])

    report_lines.append(f"\n### Dark Skin (FST IV-VI)")

    dark_fst = [4, 5, 6]
    dark_aurocs = [fairness_results.auroc_per_fst[fst] for fst in dark_fst if fst in fairness_results.auroc_per_fst]
    if dark_aurocs:
        report_lines.extend([
            f"- Average AUROC: {np.mean(dark_aurocs):.4f}",
            f"- Min AUROC: {np.min(dark_aurocs):.4f}",
            f"- Max AUROC: {np.max(dark_aurocs):.4f}"
        ])

    report_lines.extend([
        f"\n## Analysis",
        f"\n### Findings",
        f"1. The model shows {'significant' if fairness_results.auroc_gap > 0.1 else 'moderate'} "
        f"performance disparities across Fitzpatrick skin types.",
        f"2. AUROC gap (max-min): {fairness_results.auroc_gap:.4f}",
        f"3. Light-dark skin gap: {fairness_results.auroc_gap_light_dark:.4f}",
        f"   - {'Light skin shows higher performance' if fairness_results.auroc_gap_light_dark > 0 else 'Dark skin shows higher performance'}",
        f"\n### Recommendations",
        f"1. {'Consider fairness interventions (data augmentation, reweighting, adversarial debiasing)' if fairness_results.auroc_gap > 0.1 else 'Performance is relatively fair, but monitoring is recommended'}",
        f"2. Focus on {'FST ' + str(min(fairness_results.auroc_per_fst, key=fairness_results.auroc_per_fst.get)) + ' which shows lowest performance'}",
        f"3. Implement calibration techniques to improve ECE scores",
        f"\n## Visualizations",
        f"\nSee generated figures in `experiments/baseline/figures/`:",
        f"- ROC curves by FST",
        f"- Calibration curves by FST",
        f"- Confusion matrices by FST",
        f"- Fairness gap comparisons",
        f"\n---",
        f"\n*Report generated by HOLLOWED_EYES fairness evaluation framework*"
    ])

    report_content = "\n".join(report_lines)

    with open(save_path, 'w') as f:
        f.write(report_content)

    print(f"\nFairness report saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Comprehensive fairness evaluation')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['resnet50', 'efficientnet_b3', 'efficientnet_b4', 'inception_v3'],
        help='Model architecture'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/baseline_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for evaluation'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='experiments/baseline/reports',
        help='Directory to save evaluation outputs'
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print("FAIRNESS EVALUATION FRAMEWORK")
    print("="*80)
    print(f"\nModel: {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")

    # Load configuration
    config = load_config(args.config)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create model
    print(f"\nCreating {args.model} model...")
    model = create_model(args.model, config['model']['num_classes'])

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Checkpoint loaded successfully")

    # Create test loader
    print("\nLoading test dataset...")
    test_loader = create_dummy_test_loader(config)
    print(f"Test batches: {len(test_loader)}")

    # Evaluate model
    print("\n" + "="*80)
    print("EVALUATING MODEL")
    print("="*80)

    test_results = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=args.device,
        num_classes=config['model']['num_classes']
    )

    fairness_results = test_results['fairness_results']

    # Print summary
    print("\n" + fairness_results.summary())

    # Save JSON results
    results_path = output_dir / f'{args.model}_fairness_results.json'
    with open(results_path, 'w') as f:
        json.dump(fairness_results.to_dict(), f, indent=4)
    print(f"\nJSON results saved to {results_path}")

    # Generate comprehensive report
    report_path = output_dir / f'{args.model}_fairness_report.md'
    generate_fairness_report(
        model_name=args.model,
        fairness_results=fairness_results,
        save_path=report_path
    )

    # Generate visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    figures_dir = Path('experiments/baseline/figures')
    visualizer = FairnessVisualizer(save_dir=str(figures_dir))

    visualizer.plot_roc_curves_per_fst(
        y_true=test_results['y_true'],
        y_prob=test_results['y_prob'],
        fst_labels=test_results['fst_labels'],
        target_class=0,
        save_name=f'{args.model}_roc_curves_per_fst.png'
    )

    visualizer.plot_calibration_curves_per_fst(
        y_true=test_results['y_true'],
        y_prob=test_results['y_prob'],
        fst_labels=test_results['fst_labels'],
        save_name=f'{args.model}_calibration_curves_per_fst.png'
    )

    visualizer.plot_confusion_matrices_per_fst(
        y_true=test_results['y_true'],
        y_pred=test_results['y_pred'],
        fst_labels=test_results['fst_labels'],
        save_name=f'{args.model}_confusion_matrices_per_fst.png'
    )

    visualizer.plot_fairness_gaps(
        auroc_per_fst=fairness_results.auroc_per_fst,
        save_name=f'{args.model}_fairness_gaps.png'
    )

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\nOutputs saved to:")
    print(f"  - Results: {results_path}")
    print(f"  - Report: {report_path}")
    print(f"  - Figures: {figures_dir}")


if __name__ == "__main__":
    main()
