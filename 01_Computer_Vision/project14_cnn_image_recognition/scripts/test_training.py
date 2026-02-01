#!/usr/bin/env python3
"""
Quick training test for CIFAR-10 CNN classifier using processed data
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from cnn_classifier import CNNImageClassifier

def main():
    """Test training with processed CIFAR-10 data"""
    print("Testing CNN training with processed CIFAR-10 data...")

    # Initialize classifier
    classifier = CNNImageClassifier(num_classes=10)

    # Create and compile model
    print("Creating model...")
    classifier.create_model(use_transfer_learning=True)
    classifier.compile_model(learning_rate=0.001)

    # Setup data generators
    print("Setting up data generators...")
    train_gen, val_gen, test_gen = classifier.get_data_generators(
        train_dir='data/processed/train',
        val_dir='data/processed/val',
        test_dir='data/processed/test',
        batch_size=32
    )

    print("Data generators created successfully!")
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Test samples: {test_gen.samples if test_gen else 0}")

    # Quick training test (just 1 epoch to verify everything works)
    print("Starting quick training test (1 epoch)...")
    history = classifier.train(train_gen, val_gen, epochs=1)

    print("Training completed successfully!")
    print(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

    # Quick evaluation
    print("Evaluating on test set...")
    results = classifier.evaluate(test_gen)
    print(f"Test loss: {results['loss']:.4f}")
    print(f"Test accuracy: {results['accuracy']:.4f}")

    print("\n✅ CIFAR-10 CNN training pipeline is working correctly!")
    print("The project has been successfully upgraded from empty data to real CIFAR-10 dataset.")

if __name__ == "__main__":
    main()