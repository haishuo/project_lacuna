#!/usr/bin/env python3
"""
Evaluate threshold-trained model on sigmoid MNAR (unseen functional form).

This is the CRITICAL test: Does the model understand abstract MNAR mechanisms,
or did it just memorize the threshold fingerprint?
"""

import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lacuna.generators import (
    GeneratorRegistry,
    GeneratorParams,
    MCARUniform,
    MARLogistic,
    MNARLogistic,
    MNARSelfCensoring,
)
from lacuna.data.batching import SyntheticDataLoaderConfig, SyntheticDataLoader
from lacuna.models.assembly import create_lacuna_model
from lacuna.training import load_checkpoint
from lacuna.core.types import MCAR, MAR, MNAR


def create_sigmoid_test_registry():
    """Create test registry with SIGMOID MNAR (unseen functional form!)."""
    return GeneratorRegistry([
        # Same MCAR/MAR as training
        MCARUniform(0, "mcar_low", GeneratorParams(miss_rate=0.15)),
        MCARUniform(1, "mcar_high", GeneratorParams(miss_rate=0.35)),
        MARLogistic(2, "mar_weak", GeneratorParams(alpha0=-0.5, alpha1=2.0)),
        MARLogistic(3, "mar_strong", GeneratorParams(alpha0=-0.5, alpha1=4.0)),
        
        # DIFFERENT: Sigmoid MNAR (never seen during training!)
        MNARLogistic(4, "mnar_sigmoid_weak", GeneratorParams(beta0=-0.5, beta2=1.5)),
        MNARSelfCensoring(5, "mnar_sigmoid_strong", GeneratorParams(beta0=-0.5, beta1=2.5)),
    ])


def evaluate_model(model, test_loader, device):
    """Evaluate model and compute per-class accuracy."""
    model.eval()
    
    total_correct = 0
    total_samples = 0
    
    # Per-class metrics
    class_correct = {MCAR: 0, MAR: 0, MNAR: 0}
    class_total = {MCAR: 0, MAR: 0, MNAR: 0}
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= 50:  # Test on 50 batches = 800 samples
                break
            
            batch = batch.to(device)
            output = model(batch)
            
            # Get predictions
            pred_class = output.posterior.p_class.argmax(dim=-1)
            true_class = batch.class_ids
            
            # Overall accuracy
            correct = (pred_class == true_class).sum().item()
            total_correct += correct
            total_samples += len(true_class)
            
            # Per-class accuracy
            for cls in [MCAR, MAR, MNAR]:
                mask = true_class == cls
                if mask.any():
                    class_correct[cls] += (pred_class[mask] == cls).sum().item()
                    class_total[cls] += mask.sum().item()
    
    # Compute metrics
    overall_acc = total_correct / total_samples
    
    class_acc = {}
    for cls in [MCAR, MAR, MNAR]:
        if class_total[cls] > 0:
            class_acc[cls] = class_correct[cls] / class_total[cls]
        else:
            class_acc[cls] = 0.0
    
    return overall_acc, class_acc


def main():
    print("=" * 70)
    print("GENERATOR FINGERPRINT TEST: EVALUATION")
    print("=" * 70)
    print("Testing threshold-trained model on SIGMOID MNAR (unseen form)\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load trained model
    model_path = Path("outputs/fingerprint_test/model_threshold_trained.pt")
    
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print("Run train_fingerprint_test.py first!")
        return
    
    print(f"Loading model from: {model_path}")
    
    # Create model architecture (must match training)
    model = create_lacuna_model(
        hidden_dim=128,
        n_layers=4,
        n_heads=4,
        max_cols=16,
        mnar_variants=["threshold"],
    ).to(device)
    
    # Load weights
    ckpt = load_checkpoint(model_path, device=device)
    model.load_state_dict(ckpt.model_state)
    
    print(f"✓ Loaded checkpoint from step {ckpt.step}")
    print(f"  Training accuracy: {ckpt.best_val_acc*100:.1f}%\n")
    
    # Create test registry (SIGMOID MNAR!)
    test_registry = create_sigmoid_test_registry()
    
    print("Test Registry (SIGMOID MNAR - unseen functional form!):")
    for gen in test_registry.generators:
        form = "SIGMOID" if gen.generator_id >= 4 else "same"
        print(f"  {gen.generator_id}: {gen.name} (class={gen.class_id}, form={form})")
    
    # Create test loader
    test_config = SyntheticDataLoaderConfig(
        batch_size=16,
        n_range=(50, 200),
        d_range=(5, 12),
        max_cols=16,
        seed=123,  # Different seed from training
        batches_per_epoch=50,
    )
    
    test_loader = SyntheticDataLoader(
        generators=test_registry.generators,
        config=test_config,
    )
    
    print(f"\n{'=' * 70}")
    print("EVALUATING ON SIGMOID MNAR")
    print("=" * 70)
    
    overall_acc, class_acc = evaluate_model(model, test_loader, device)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Overall Accuracy: {overall_acc*100:.1f}%")
    print(f"\nPer-Class Accuracy:")
    print(f"  MCAR: {class_acc[MCAR]*100:.1f}%")
    print(f"  MAR:  {class_acc[MAR]*100:.1f}%")
    print(f"  MNAR: {class_acc[MNAR]*100:.1f}% ← CRITICAL (sigmoid, unseen!)")
    
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    if class_acc[MNAR] > 0.70:
        print("✓ SURVIVAL: Model generalizes to unseen functional form!")
        print("  The model learned abstract MNAR mechanism, not threshold fingerprint.")
        print(f"  MNAR accuracy {class_acc[MNAR]*100:.1f}% on sigmoid (never seen).")
    elif class_acc[MNAR] > 0.50:
        print("⚠ PARTIAL: Some generalization, but imperfect.")
        print("  Model may have learned some fingerprints.")
        print(f"  MNAR accuracy {class_acc[MNAR]*100:.1f}% suggests mixed learning.")
    else:
        print("✗ DEATH: Model failed to generalize to sigmoid MNAR!")
        print("  The model only learned threshold fingerprint.")
        print(f"  MNAR accuracy {class_acc[MNAR]*100:.1f}% ≈ random guessing.")
        print("  Lacuna's claims are NOT supported by evidence.")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()