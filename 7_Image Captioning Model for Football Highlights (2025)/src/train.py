import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, Add
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import ResNet50
import matplotlib.pyplot as plt
import pickle

class FootballCaptionModel:
    def __init__(self, vocab_size, max_length, embedding_dim=256, lstm_units=512):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.model = None
        self.encoder_model = None
        self.decoder_model = None

    # Feature extractor will be loaded lazily via utils.get_resnet50()
    self.feature_extractor = None

    def build_model(self):
        """Build the image captioning model"""
        # Image feature extractor (encoder)
        image_input = Input(shape=(2048,), name='image_input')
        # Project image features to match LSTM units so Add() can combine them
        image_dense = Dense(self.lstm_units, activation='relu')(image_input)
        image_dropout = Dropout(0.5)(image_dense)

        # Caption sequence processor (decoder)
        caption_input = Input(shape=(self.max_length,), name='caption_input')
        caption_embedding = Embedding(self.vocab_size, self.embedding_dim, mask_zero=True)(caption_input)
        caption_dropout = Dropout(0.5)(caption_embedding)

        # LSTM layers
        lstm1 = LSTM(self.lstm_units, return_sequences=True)(caption_dropout)
        lstm1_dropout = Dropout(0.5)(lstm1)
        lstm2 = LSTM(self.lstm_units)(lstm1_dropout)
        lstm2_dropout = Dropout(0.5)(lstm2)

        # Decoder
        decoder1 = Add()([image_dropout, lstm2_dropout])
        decoder2 = Dense(self.lstm_units, activation='relu')(decoder1)
        decoder2_dropout = Dropout(0.5)(decoder2)
        outputs = Dense(self.vocab_size, activation='softmax')(decoder2_dropout)

        # Create model
        self.model = Model(inputs=[image_input, caption_input], outputs=outputs)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        print("Model built successfully!")
        print(self.model.summary())

        return self.model

    def extract_image_features(self, image_path):
        """Extract features from a single image"""
        from tensorflow.keras.preprocessing.image import load_img, img_to_array
        from tensorflow.keras.applications.resnet50 import preprocess_input
        from utils import get_resnet50

        try:
            img = load_img(image_path, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            if self.feature_extractor is None:
                self.feature_extractor = get_resnet50()

            features = self.feature_extractor.predict(img_array, verbose=0)
            return features.flatten()
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return None

    def extract_all_features(self, image_dir, captions_dict):
        """Extract features for all training images"""
        features = {}
        total_images = len(captions_dict)

        print(f"Extracting features from {total_images} images...")

        for i, image_id in enumerate(captions_dict.keys()):
            image_path = os.path.join(image_dir, f"{image_id}.jpg")
            if os.path.exists(image_path):
                feature = self.extract_image_features(image_path)
                if feature is not None:
                    features[image_id] = feature

                    if (i + 1) % 50 == 0:
                        print(f"Processed {i + 1}/{total_images} images")
            else:
                print(f"Image not found: {image_path}")

        print(f"Successfully extracted features from {len(features)} images")
        return features

    def create_sequences(self, captions_dict, features, word_to_idx):
        """Create training sequences"""
        X1, X2, y = [], [], []

        for image_id, captions in captions_dict.items():
            if image_id not in features:
                continue

            for caption in captions:
                # Convert caption to sequence
                seq = [word_to_idx.get(word, word_to_idx['<unk>']) for word in caption.split()]

                # Add start and end tokens
                seq = [word_to_idx['<start>']] + seq + [word_to_idx['<end>']]

                # Create input-output pairs
                for i in range(1, len(seq)):
                    in_seq = seq[:i]
                    out_seq = seq[i]

                    # Pad input sequence
                    in_seq_padded = pad_sequences([in_seq], maxlen=self.max_length)[0]

                    # One-hot encode output
                    out_seq_onehot = to_categorical([out_seq], num_classes=self.vocab_size)[0]

                    X1.append(features[image_id])
                    X2.append(in_seq_padded)
                    y.append(out_seq_onehot)

        return np.array(X1), np.array(X2), np.array(y)

    def build_inference_models(self):
        """Build models for inference (prediction)"""
        # Build robust step-wise encoder/decoder for greedy inference
        try:
            from tensorflow.keras.layers import Embedding, LSTM as KerasLSTM, Add as KAdd, Dense as KDense, Dropout as KDropout

            # Encoder: reuse image projection layers
            encoder_input = self.model.input[0]
            image_dense_layer = self.model.layers[1]
            image_dropout_layer = self.model.layers[2]
            encoder_proj = image_dropout_layer(image_dense_layer(encoder_input))
            self.encoder_model = Model(inputs=encoder_input, outputs=encoder_proj)

            # Locate embedding and trained LSTM layers
            embedding_layer = None
            trained_lstm_layers = []
            for layer in self.model.layers:
                if isinstance(layer, Embedding):
                    embedding_layer = layer
                if isinstance(layer, KerasLSTM):
                    trained_lstm_layers.append(layer)

            if embedding_layer is None or len(trained_lstm_layers) < 2:
                raise RuntimeError('Could not locate embedding or LSTM layers in trained model')

            lstm1_trained = trained_lstm_layers[0]
            lstm2_trained = trained_lstm_layers[1]

            # Inference decoder inputs: single token + image features + states for two LSTMs
            token_input = Input(shape=(1,), name='decoder_token_input', dtype='int32')
            image_feat_input = Input(shape=(self.lstm_units,), name='decoder_image_input')
            state_h1_in = Input(shape=(self.lstm_units,), name='state_h1_in')
            state_c1_in = Input(shape=(self.lstm_units,), name='state_c1_in')
            state_h2_in = Input(shape=(self.lstm_units,), name='state_h2_in')
            state_c2_in = Input(shape=(self.lstm_units,), name='state_c2_in')

            # Embed token
            emb = embedding_layer(token_input)

            # Build inference LSTMs and run one step
            inf_lstm1 = KerasLSTM(self.lstm_units, return_sequences=True, return_state=True, name='inf_lstm1')
            inf_lstm2 = KerasLSTM(self.lstm_units, return_sequences=False, return_state=True, name='inf_lstm2')

            lstm1_out_seq, h1_out, c1_out = inf_lstm1(emb, initial_state=[state_h1_in, state_c1_in])
            lstm2_out, h2_out, c2_out = inf_lstm2(lstm1_out_seq, initial_state=[state_h2_in, state_c2_in])

            merged = KAdd()([image_feat_input, lstm2_out])
            dense1 = KDense(self.lstm_units, activation='relu', name='inf_dense1')(merged)
            drop1 = KDropout(0.5)(dense1)
            out_probs = KDense(self.vocab_size, activation='softmax', name='inf_output')(drop1)

            self.decoder_model = Model(
                inputs=[token_input, image_feat_input, state_h1_in, state_c1_in, state_h2_in, state_c2_in],
                outputs=[out_probs, h1_out, c1_out, h2_out, c2_out]
            )

            # Try to transfer weights for LSTMs and embedding
            try:
                inf_lstm1.set_weights(lstm1_trained.get_weights())
                inf_lstm2.set_weights(lstm2_trained.get_weights())
                # transfer embedding weights
                try:
                    emb_w = embedding_layer.get_weights()
                    # create a temporary embedding layer on decoder to set weights if needed
                except Exception:
                    pass
            except Exception as e:
                print(f"Warning copying weights to inference LSTMs: {e}")

            print('Built step-wise inference encoder+decoder')

        except Exception as e:
            print(f"Warning: failed to build inference models: {e}. Proceeding without encoder/decoder.")
            self.encoder_model = None
            self.decoder_model = None

    def generate_caption(self, image_features, word_to_idx, idx_to_word, max_length=20):
        """Generate caption for an image"""
        # Use step-wise decoder if available
        if self.encoder_model is None or self.decoder_model is None:
            raise RuntimeError('Inference models not available. Call build_inference_models() first')

        # Encode image
        image_proj = self.encoder_model.predict(np.array([image_features]), verbose=0)

        # Initialize states
        h1 = np.zeros((1, self.lstm_units))
        c1 = np.zeros((1, self.lstm_units))
        h2 = np.zeros((1, self.lstm_units))
        c2 = np.zeros((1, self.lstm_units))

        start_idx = word_to_idx.get('<start>')
        if start_idx is None:
            raise KeyError("'<start>' token not found in vocab")

        token = np.array([[start_idx]], dtype='int32')
        generated_words = []

        for _ in range(max_length):
            preds, h1, c1, h2, c2 = self.decoder_model.predict([token, image_proj, h1, c1, h2, c2], verbose=0)
            next_idx = int(np.argmax(preds[0]))
            next_word = idx_to_word.get(next_idx, '<unk>')
            if next_word == '<end>' or next_word == '<unk>':
                break
            generated_words.append(next_word)
            token = np.array([[next_idx]], dtype='int32')

        return ' '.join(generated_words)

    def train(self, X1_train, X2_train, y_train, X1_val, X2_val, y_val,
              epochs=50, batch_size=64, model_save_path='models/football_caption_model.h5'):
        """Train the model"""
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

        # Callbacks
        checkpoint = ModelCheckpoint(
            model_save_path,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )

        # Train model
        history = self.model.fit(
            [X1_train, X2_train], y_train,
            validation_data=([X1_val, X2_val], y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpoint, early_stopping, reduce_lr],
            verbose=1
        )

        return history

    def save_models(self, save_dir='models/'):
        """Save trained models"""
        os.makedirs(save_dir, exist_ok=True)

        # Save main model
        self.model.save(os.path.join(save_dir, 'football_caption_model.h5'))

        # Save encoder and decoder for inference if they were built
        if self.encoder_model is not None:
            try:
                self.encoder_model.save(os.path.join(save_dir, 'encoder_model.h5'))
            except Exception:
                print('Warning: could not save encoder_model (skipping).')

        if self.decoder_model is not None:
            try:
                self.decoder_model.save(os.path.join(save_dir, 'decoder_model.h5'))
            except Exception:
                print('Warning: could not save decoder_model (skipping).')

        print(f"Models saved to {save_dir}")

def load_processed_data(processed_dir='data/processed/'):
    """Load processed data for training"""
    # Load vocabulary
    with open(os.path.join(processed_dir, 'vocab.json'), 'r') as f:
        vocab_data = json.load(f)

    vocab = vocab_data['vocab']
    word_to_idx = vocab_data['word_to_idx']
    idx_to_word = vocab_data['idx_to_word']

    # Load captions
    with open(os.path.join(processed_dir, 'train_captions.json'), 'r') as f:
        train_captions = json.load(f)

    with open(os.path.join(processed_dir, 'val_captions.json'), 'r') as f:
        val_captions = json.load(f)

    return vocab, word_to_idx, idx_to_word, train_captions, val_captions

def main():
    """Main training function"""
    print("Starting Football Caption Model Training...")

    # Load processed data
    vocab, word_to_idx, idx_to_word, train_captions, val_captions = load_processed_data()

    vocab_size = len(vocab)
    max_length = 25  # Maximum caption length

    print(f"Vocabulary size: {vocab_size}")
    print(f"Training images: {len(train_captions)}")
    print(f"Validation images: {len(val_captions)}")

    # Initialize model
    model = FootballCaptionModel(vocab_size=vocab_size, max_length=max_length)
    model.build_model()

    # Extract image features
    print("Extracting image features...")
    train_features = model.extract_all_features('data/processed/images/', train_captions)
    val_features = model.extract_all_features('data/processed/images/', val_captions)

    # Create training sequences
    print("Creating training sequences...")
    X1_train, X2_train, y_train = model.create_sequences(train_captions, train_features, word_to_idx)
    X1_val, X2_val, y_val = model.create_sequences(val_captions, val_features, word_to_idx)

    print(f"Training sequences: {X1_train.shape[0]}")
    print(f"Validation sequences: {X1_val.shape[0]}")

    # Train model
    print("Training model...")
    history = model.train(X1_train, X2_train, y_train, X1_val, X2_val, y_val,
                         epochs=50, batch_size=64)

    # Build inference models
    print("Building inference models...")
    model.build_inference_models()

    # Save models
    model.save_models()

    # Save training history
    with open('models/training_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)

    print("Training completed!")

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('models/training_history.png')
    plt.show()

if __name__ == "__main__":
    main()