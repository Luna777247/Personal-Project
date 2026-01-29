import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, Add
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import pickle

class FootballCaptionModel:
    def __init__(self, vocab_size, max_length, embedding_dim=256, lstm_units=256):
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.model = None

    def build_model(self):
        """Build the CNN-LSTM model architecture"""

        # Image feature extractor (ResNet50)
        resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')
        for layer in resnet.layers:
            layer.trainable = False

        # Image input
        image_input = Input(shape=(2048,))
        image_dense = Dense(self.embedding_dim, activation='relu')(image_input)
        image_dropout = Dropout(0.5)(image_dense)
        image_output = Dense(self.lstm_units, activation='relu')(image_dropout)

        # Text input
        text_input = Input(shape=(self.max_length,))
        text_embedding = Embedding(self.vocab_size, self.embedding_dim, mask_zero=True)(text_input)
        text_dropout = Dropout(0.5)(text_embedding)
        text_lstm = LSTM(self.lstm_units)(text_dropout)

        # Decoder
        decoder_input = Add()([image_output, text_lstm])
        decoder_dense1 = Dense(self.lstm_units, activation='relu')(decoder_input)
        decoder_dropout = Dropout(0.5)(decoder_dense1)
        decoder_output = Dense(self.vocab_size, activation='softmax')(decoder_dropout)

        # Define model
        self.model = Model(inputs=[image_input, text_input], outputs=decoder_output)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return self.model

    def build_inference_model(self):
        """Build model for inference (caption generation)"""

        # Encoder model
        encoder_input = self.model.input[0]  # image input
        encoder_output = self.model.layers[3].output  # image features after processing
        encoder_model = Model(encoder_input, encoder_output)

        # Decoder model
        decoder_input_text = Input(shape=(self.max_length,))
        decoder_embedding = self.model.layers[4](decoder_input_text)
        decoder_lstm = self.model.layers[6](decoder_embedding)

        # Get decoder states
        decoder_state_input_h = Input(shape=(self.lstm_units,))
        decoder_state_input_c = Input(shape=(self.lstm_units,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        decoder_lstm_output, state_h, state_c = LSTM(self.lstm_units, return_sequences=True, return_state=True)(
            decoder_embedding, initial_state=decoder_states_inputs
        )

        decoder_states = [state_h, state_c]
        decoder_dense = self.model.layers[8](decoder_lstm_output)
        decoder_outputs = self.model.layers[9](decoder_dense)

        decoder_model = Model(
            [decoder_input_text] + decoder_states_inputs,
            [decoder_outputs] + decoder_states
        )

        return encoder_model, decoder_model

    def generate_caption(self, image_feature, tokenizer, max_length):
        """Generate caption for an image"""

        # Start with start token
        in_text = '<start>'

        for i in range(max_length):
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            sequence = pad_sequences([sequence], maxlen=max_length)

            # Predict next word
            yhat = self.model.predict([image_feature, sequence], verbose=0)
            yhat = np.argmax(yhat)

            # Convert to word
            word = tokenizer.index_word.get(yhat, '<unk>')

            # Stop if end token or unknown
            if word is None or word == '<end>':
                break

            in_text += ' ' + word

            # Stop if caption too long
            if len(in_text.split()) > max_length:
                break

        return in_text.replace('<start> ', '').replace(' <end>', '')

    def save_model(self, filepath):
        """Save the trained model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load a trained model"""
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")

def create_model(vocab_size, max_length):
    """Factory function to create model"""
    model = FootballCaptionModel(vocab_size, max_length)
    model.build_model()
    return model