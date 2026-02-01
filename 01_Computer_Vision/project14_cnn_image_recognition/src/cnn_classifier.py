import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class CNNImageClassifier:
    def __init__(self, input_shape=(224, 224, 3), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None

    def create_model(self, use_transfer_learning=True):
        """Create CNN model with optional transfer learning"""
        if use_transfer_learning:
            # Use EfficientNetB0 as base model
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
            base_model.trainable = False  # Freeze base model initially

            self.model = models.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(self.num_classes, activation='softmax')
            ])
        else:
            # Custom CNN architecture
            self.model = models.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),

                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),

                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),

                layers.Conv2D(256, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),

                layers.Flatten(),
                layers.Dense(512, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(self.num_classes, activation='softmax')
            ])

        return self.model

    def get_data_generators(self, train_dir, val_dir=None, test_dir=None,
                          batch_size=32, augmentation=True):
        """Create data generators with augmentation"""

        if augmentation:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                zoom_range=0.2,
                fill_mode='nearest',
                validation_split=0.2 if val_dir is None else 0
            )
        else:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=0.2 if val_dir is None else 0
            )

        test_datagen = ImageDataGenerator(rescale=1./255)

        # Training generator
        if val_dir is None:
            train_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=self.input_shape[:2],
                batch_size=batch_size,
                class_mode='categorical',
                subset='training'
            )

            validation_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=self.input_shape[:2],
                batch_size=batch_size,
                class_mode='categorical',
                subset='validation'
            )
        else:
            train_generator = train_datagen.flow_from_directory(
                train_dir,
                target_size=self.input_shape[:2],
                batch_size=batch_size,
                class_mode='categorical'
            )

            validation_generator = test_datagen.flow_from_directory(
                val_dir,
                target_size=self.input_shape[:2],
                batch_size=batch_size,
                class_mode='categorical'
            )

        test_generator = None
        if test_dir:
            test_generator = test_datagen.flow_from_directory(
                test_dir,
                target_size=self.input_shape[:2],
                batch_size=batch_size,
                class_mode='categorical',
                shuffle=False
            )

        return train_generator, validation_generator, test_generator

    def compile_model(self, learning_rate=0.001):
        """Compile the model with optimizer and loss function"""
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3)]
        )

    def train(self, train_generator, validation_generator, epochs=50,
             callbacks_list=None):
        """Train the model"""
        if callbacks_list is None:
            callbacks_list = [
                callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=10,
                    restore_best_weights=True
                ),
                callbacks.ReduceLROnPlateau(
                    monitor='val_accuracy',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7
                ),
                callbacks.ModelCheckpoint(
                    'models/best_model.h5',
                    monitor='val_accuracy',
                    save_best_only=True
                )
            ]

        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks_list
        )

        return history

    def evaluate(self, test_generator):
        """Evaluate model performance"""
        if test_generator is None:
            return None

        # Get loss and accuracy from model.evaluate
        eval_results = self.model.evaluate(test_generator, verbose=0)
        loss = eval_results[0] if isinstance(eval_results, list) else eval_results['loss']
        accuracy = eval_results[1] if isinstance(eval_results, list) else eval_results['accuracy']

        # Get predictions
        predictions = self.model.predict(test_generator, verbose=0)
        y_pred = np.argmax(predictions, axis=1)
        y_true = test_generator.classes

        # Classification report
        class_names = list(test_generator.class_indices.keys())
        report = classification_report(y_true, y_pred, target_names=class_names)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        return {
            'loss': loss,
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'class_names': class_names
        }

    def plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Accuracy plot
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()

        # Loss plot
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()

        plt.tight_layout()
        plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

    def convert_to_tflite(self, model_path='models/best_model.h5',
                         tflite_path='models/model.tflite'):
        """Convert model to TensorFlow Lite format"""
        # Load the best model
        model = tf.keras.models.load_model(model_path)

        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        # Optimization settings
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]  # Use float16 for better performance

        tflite_model = converter.convert()

        # Save the model
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)

        print(f"Model converted and saved to {tflite_path}")
        return tflite_path

    def quantize_model(self, tflite_path='models/model.tflite',
                      quantized_path='models/model_quantized.tflite'):
        """Apply post-training quantization"""
        # Load TFLite model
        with open(tflite_path, 'rb') as f:
            tflite_model = f.read()

        # Apply quantization
        converter = tf.lite.TFLiteConverter.from_saved_model('models/saved_model')
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Quantize to int8
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        quantized_model = converter.convert()

        # Save quantized model
        with open(quantized_path, 'wb') as f:
            f.write(quantized_model)

        print(f"Quantized model saved to {quantized_path}")
        return quantized_path

def main():
    """Main training pipeline"""
    # Initialize classifier
    classifier = CNNImageClassifier(num_classes=10)  # Adjust based on your dataset

    # Create model
    model = classifier.create_model(use_transfer_learning=True)
    classifier.compile_model()

    # Setup data generators (you'll need to provide actual data directories)
    # train_gen, val_gen, test_gen = classifier.get_data_generators(
    #     train_dir='data/train',
    #     val_dir='data/val',
    #     test_dir='data/test'
    # )

    # Train model
    # history = classifier.train(train_gen, val_gen, epochs=50)

    # Evaluate
    # results = classifier.evaluate(test_gen)

    # Convert to TFLite
    # classifier.convert_to_tflite()

    print("CNN Image Classification pipeline ready!")
    print("Uncomment the training code and provide your dataset paths to start training.")

if __name__ == "__main__":
    main()