import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Streamlit is optional for the web demo; import only if available
try:
    import streamlit as st
except Exception:
    st = None
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

class FootballCaptionDemo:
    def __init__(self, model_path='models/football_caption_model.h5',
                 vocab_path='data/processed/vocab.json',
                 max_length=25):
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.max_length = max_length

        # Load vocabulary
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
        self.vocab = vocab_data['vocab']
        self.word_to_idx = vocab_data['word_to_idx']
        self.idx_to_word = vocab_data['idx_to_word']

        # Load model
        self.model = load_model(model_path)
        print(f"Model loaded from {model_path}")

        # Build inference models
        self.build_inference_models()

    def build_inference_models(self):
        """Build encoder and decoder models for inference"""
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, Add

        # Encoder model
        encoder_input = self.model.input[0]  # image input
        encoder_dense = self.model.layers[1](encoder_input)  # image dense
        encoder_dropout = self.model.layers[2](encoder_dense)  # image dropout

        self.encoder_model = Model(inputs=encoder_input, outputs=encoder_dropout)

        # Decoder model
        decoder_input_h = Input(shape=(512,))  # LSTM units
        decoder_input_c = Input(shape=(512,))
        decoder_states_inputs = [decoder_input_h, decoder_input_c]

        decoder_embedding = self.model.layers[3](self.model.input[1])  # caption embedding
        decoder_dropout1 = self.model.layers[4](decoder_embedding)  # caption dropout

        decoder_lstm1 = self.model.layers[5](decoder_dropout1)  # lstm1
        decoder_dropout2 = self.model.layers[6](decoder_lstm1)  # lstm1 dropout

        decoder_lstm2, state_h, state_c = LSTM(512, return_sequences=False, return_state=True)(decoder_dropout2)
        decoder_states = [state_h, state_c]

        decoder_dropout3 = Dropout(0.5)(decoder_lstm2)  # lstm2 dropout

        # For inference, we need to handle the Add layer differently
        # We'll create a simpler decoder that takes image features as input
        image_features_input = Input(shape=(256,))  # embedding dim
        decoder_add = Add()([image_features_input, decoder_dropout3])
        decoder_dense1 = Dense(512, activation='relu')(decoder_add)
        decoder_dropout4 = Dropout(0.5)(decoder_dense1)
        decoder_outputs = Dense(len(self.vocab), activation='softmax')(decoder_dropout4)

        self.decoder_model = Model(
            inputs=[self.model.input[1], image_features_input, decoder_states_inputs],
            outputs=[decoder_outputs, decoder_states]
        )

    def extract_image_features(self, image_path):
        """Extract features from an image using ResNet50"""
        from tensorflow.keras.preprocessing.image import load_img, img_to_array
        from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50

        feature_extractor = ResNet50(weights='imagenet', include_top=False, pooling='avg')

        try:
            img = load_img(image_path, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            features = feature_extractor.predict(img_array, verbose=0)
            return features.flatten()
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return None

    def generate_caption(self, image_features):
        """Generate caption for image features"""
        # Encode image
        image_features_encoded = self.encoder_model.predict(np.array([image_features]), verbose=0)

        # Start with <start> token
        start_token = self.word_to_idx['<start>']
        caption = [start_token]

        for _ in range(self.max_length):
            # Prepare input sequence
            sequence = pad_sequences([caption], maxlen=self.max_length)

            # Predict next word
            yhat, h, c = self.decoder_model.predict(
                [sequence, image_features_encoded,
                 [np.zeros((1, 512)), np.zeros((1, 512))]], verbose=0
            )

            # Get word with highest probability
            predicted_word_idx = np.argmax(yhat[0])

            # Convert to word
            predicted_word = self.idx_to_word.get(predicted_word_idx, '<unk>')

            # Stop if end token or unknown
            if predicted_word == '<end>' or predicted_word == '<unk>':
                break

            caption.append(predicted_word_idx)

        # Convert indices to words (skip <start>)
        caption_words = [self.idx_to_word.get(idx, '<unk>') for idx in caption[1:]]

        return ' '.join(caption_words)

    def predict_from_image_path(self, image_path):
        """Generate caption from image path"""
        features = self.extract_image_features(image_path)
        if features is not None:
            caption = self.generate_caption(features)
            return caption
        return "Error processing image"

    def predict_from_uploaded_file(self, uploaded_file):
        """Generate caption from uploaded file"""
        try:
            # Save uploaded file temporarily
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Generate caption
            caption = self.predict_from_image_path(temp_path)

            # Clean up
            os.remove(temp_path)

            return caption
        except Exception as e:
            return f"Error: {str(e)}"

def create_streamlit_app():
    """Create Streamlit web application"""
    if st is None:
        raise RuntimeError("Streamlit is not installed in this environment. To run the web demo install streamlit (pip install streamlit).")

    st.set_page_config(
        page_title="Football Caption Generator",
        page_icon="⚽",
        layout="wide"
    )

    st.title("⚽ Football Image Caption Generator")
    st.markdown("Upload a football image and get an AI-generated caption describing the action!")

    # Initialize model
    try:
        demo = FootballCaptionDemo()
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a football image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload an image of a football match scene"
    )

    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uploaded Image")
            st.image(uploaded_file, use_column_width=True)

        with col2:
            st.subheader("Generated Caption")

            if st.button("Generate Caption", type="primary"):
                with st.spinner("Generating caption..."):
                    caption = demo.predict_from_uploaded_file(uploaded_file)

                if caption.startswith("Error"):
                    st.error(caption)
                else:
                    st.success("Caption generated!")
                    st.markdown(f"**{caption}**")

                    # Add some styling
                    st.markdown("""
                    <style>
                    .caption-box {
                        background-color: #f0f2f6;
                        padding: 20px;
                        border-radius: 10px;
                        border-left: 5px solid #ff4b4b;
                        font-size: 18px;
                        font-weight: bold;
                        color: #1f1f1f;
                    }
                    </style>
                    """, unsafe_allow_html=True)

                    st.markdown(f'<div class="caption-box">{caption}</div>', unsafe_allow_html=True)

    # Sample images section
    st.markdown("---")
    st.subheader("Try with Sample Images")

    sample_images = [
        "data/processed/images/sample1.jpg",
        "data/processed/images/sample2.jpg",
        "data/processed/images/sample3.jpg"
    ]

    # Filter existing sample images
    existing_samples = [img for img in sample_images if os.path.exists(img)]

    if existing_samples:
        cols = st.columns(len(existing_samples))

        for i, (col, img_path) in enumerate(zip(cols, existing_samples)):
            with col:
                try:
                    img = Image.open(img_path)
                    st.image(img, use_column_width=True, caption=f"Sample {i+1}")

                    if st.button(f"Generate Caption for Sample {i+1}"):
                        with st.spinner("Generating caption..."):
                            caption = demo.predict_from_image_path(img_path)

                        if not caption.startswith("Error"):
                            st.success(f"**{caption}**")
                        else:
                            st.error(caption)
                except Exception as e:
                    st.error(f"Error loading sample image: {e}")
    else:
        st.info("No sample images found. Upload your own images to get started!")

    # Model information
    st.markdown("---")
    st.subheader("About the Model")
    st.markdown("""
    This AI model uses a deep learning architecture combining:
    - **ResNet50** for image feature extraction
    - **LSTM** networks for sequence generation
    - **Attention mechanism** for better caption quality

    The model was trained on a diverse dataset of football match images and can generate descriptive captions for various football scenes including goals, tackles, celebrations, and more.
    """)

    # Performance metrics (if available)
    try:
        with open('results/evaluation_results.json', 'r') as f:
            metrics = json.load(f)

        st.subheader("Model Performance")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("BLEU-4 Score", f"{metrics.get('BLEU-4', 0):.3f}")

        with col2:
            st.metric("METEOR Score", f"{metrics.get('METEOR', 0):.3f}")

        with col3:
            st.metric("ROUGE-L Score", f"{metrics.get('ROUGE-L', 0):.3f}")

    except:
        pass

def run_console_demo():
    """Run console-based demo"""
    print("Football Caption Generator - Console Demo")
    print("=" * 50)

    try:
        demo = FootballCaptionDemo()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    while True:
        image_path = input("\nEnter image path (or 'quit' to exit): ").strip()

        if image_path.lower() in ['quit', 'q', 'exit']:
            break

        if not os.path.exists(image_path):
            print("Image file not found. Please check the path.")
            continue

        print("Generating caption...")
        caption = demo.predict_from_image_path(image_path)

        if caption.startswith("Error"):
            print(f"Error: {caption}")
        else:
            print(f"\nGenerated Caption: {caption}")
            print("-" * 50)

def main():
    """Main function"""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--console":
        run_console_demo()
    else:
        # Check if running in Streamlit
        try:
            create_streamlit_app()
        except NameError:
            # Not in Streamlit environment, run console demo
            run_console_demo()

if __name__ == "__main__":
    main()