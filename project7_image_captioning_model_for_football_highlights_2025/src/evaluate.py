import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Optional metric libraries (guard imports so evaluation can run without them)
try:
    from nltk.translate.bleu_score import corpus_bleu
    from nltk.translate.meteor_score import meteor_score
    HAVE_NLTK = True
except Exception:
    HAVE_NLTK = False

try:
    from rouge_score import rouge_scorer
    HAVE_ROUGE = True
except Exception:
    HAVE_ROUGE = False

import tensorflow as tf


def _load_vocab(processed_dir='data/processed'):
    vocab_path = os.path.join(processed_dir, 'vocab.json')
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    return vocab_data['vocab'], vocab_data['word_to_idx'], vocab_data['idx_to_word']


def _load_captions(processed_dir='data/processed', split='test'):
    captions_path = os.path.join(processed_dir, f'{split}_captions.json')
    with open(captions_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _safe_predict_caption(evaluator, image_features, **kwargs):
    """Try multiple generation functions in order to be robust to model types."""
    # Preferred: FootballCaptionModel.generate_caption
    try:
        return evaluator.generate_caption(image_features, **kwargs)
    except Exception:
        pass

    # Next: evaluator may have a decoder_model + encoder_model API
    try:
        # encode image
        img_enc = evaluator.encoder_model.predict(np.array([image_features]), verbose=0)
        # fallback simple greedy loop that expects evaluator.decoder_model as token-step decoder
        if evaluator.decoder_model is None:
            raise RuntimeError('No decoder_model available')

        # prepare token / state shapes dynamically
        word_to_idx = kwargs.get('word_to_idx')
        idx_to_word = kwargs.get('idx_to_word')
        max_length = kwargs.get('max_length', 20)

        start_idx = word_to_idx.get('<start>')
        if start_idx is None:
            raise KeyError("'<start>' token not found in vocab")

        token = np.array([[start_idx]])

        # try to infer state inputs from model
        # We'll attempt a simple loop that passes zeros for unknown states
        states = [np.zeros((1, s.shape[-1])) for s in getattr(evaluator, 'initial_states_shapes', [])]

        generated = []
        for _ in range(max_length):
            try:
                preds_and_states = evaluator.decoder_model.predict([token, img_enc, *states], verbose=0)
            except Exception:
                # try different input ordering
                preds_and_states = evaluator.decoder_model.predict([token, img_enc], verbose=0)

            # first output assumed to be probs
            preds = preds_and_states[0]
            next_idx = int(np.argmax(preds[0]))
            next_word = idx_to_word.get(next_idx, '<unk>')
            if next_word in ('<end>', '<unk>'):
                break
            generated.append(next_word)
            token = np.array([[next_idx]])

        return ' '.join(generated)
    except Exception:
        # last fallback: return empty string
        return ''


class FootballCaptionEvaluator:
    """Consolidated evaluator with robust model loading and guarded metrics.

    Usage:
      evaluator = FootballCaptionEvaluator(model_path)
      results, captions = evaluator.evaluate(max_samples=50)
    """

    def __init__(self, model_path='models/football_caption_model.h5', processed_dir='data/processed', max_length=25):
        self.model_path = model_path
        self.processed_dir = processed_dir
        self.max_length = max_length

        # Load vocab and captions
        self.vocab, self.word_to_idx, self.idx_to_word = _load_vocab(processed_dir)
        self.test_captions = _load_captions(processed_dir, 'test')

        # Model placeholders
        self.model = None
        self.encoder_model = None
        self.decoder_model = None

        # Try robust loading: prefer loading a Keras model; if that fails, try to import project model class and load weights
        self._robust_model_load()

    def _robust_model_load(self):
        # 1) Try to load full model (may fail across TF versions)
        try:
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            print(f'Loaded full model from {self.model_path}')
            # attempt to attach encoder/decoder if present
            try:
                # If training code saved encoder/decoder separately, try loading them
                encoder_path = os.path.join(os.path.dirname(self.model_path), 'encoder_model.h5')
                decoder_path = os.path.join(os.path.dirname(self.model_path), 'decoder_model.h5')
                if os.path.exists(encoder_path):
                    self.encoder_model = tf.keras.models.load_model(encoder_path, compile=False)
                if os.path.exists(decoder_path):
                    self.decoder_model = tf.keras.models.load_model(decoder_path, compile=False)
            except Exception:
                # non-fatal
                pass
            return
        except Exception as e:
            print(f'Could not load full model with load_model(): {e}. Will try building model class and loading weights.')

        # 2) Try to import the project's model class and load weights
        try:
            from src.train import FootballCaptionModel

            vocab_size = len(self.vocab)
            model_wrapper = FootballCaptionModel(vocab_size=vocab_size, max_length=self.max_length)
            model_wrapper.build_model()

            # load weights into wrapper.model if possible
            try:
                model_wrapper.model.load_weights(self.model_path)
                print(f'Loaded weights into FootballCaptionModel from {self.model_path}')
            except Exception as e2:
                print(f'Failed to load weights into model wrapper: {e2}')

            # Build inference models and attach helpers
            model_wrapper.build_inference_models()
            self.model = model_wrapper.model
            self.encoder_model = getattr(model_wrapper, 'encoder_model', None)
            self.decoder_model = getattr(model_wrapper, 'decoder_model', None)

            # If wrapper provides convenient methods, attach them for use by this evaluator
            self.generate_caption = lambda feats, **kw: model_wrapper.generate_caption(feats, self.word_to_idx, self.idx_to_word, max_length=self.max_length)
            self.extract_image_features = lambda p: model_wrapper.extract_image_features(p)

            # If decoder_model expects specific initial state shapes, record template shapes
            if self.decoder_model is not None:
                # try to infer expected state shapes (common pattern: multiple state inputs after token & image)
                try:
                    input_shapes = [i.shape for i in self.decoder_model.inputs]
                    # skip token and image inputs
                    if len(input_shapes) > 2:
                        self.initial_states_shapes = [s for s in input_shapes[2:]]
                    else:
                        self.initial_states_shapes = []
                except Exception:
                    self.initial_states_shapes = []

            print('Model wrapper ready for inference')
            return
        except Exception as e:
            print(f'Failed to import/build project model: {e}')

        raise RuntimeError('Unable to load or build a model for evaluation')

    def evaluate(self, image_dir='data/processed/images/', max_samples=100):
        """Evaluate the model on the test split and compute available metrics.

        Returns (evaluation_results_dict, generated_captions_dict)
        """
        generated_captions = {}
        processed = 0

        for image_id in list(self.test_captions.keys())[:max_samples]:
            image_path = os.path.join(image_dir, f"{image_id}.jpg")
            if not os.path.exists(image_path):
                continue

            feats = self.extract_image_features(image_path)
            if feats is None:
                continue

            # Use robust generation helper
            caption = _safe_predict_caption(self, feats, word_to_idx=self.word_to_idx, idx_to_word=self.idx_to_word, max_length=self.max_length)
            generated_captions[image_id] = caption
            processed += 1

            if processed % 10 == 0:
                print(f'Processed {processed}/{min(max_samples, len(self.test_captions))} images')

        print(f'Generated captions for {len(generated_captions)} images')

        # Compute metrics (only those available)
        results = {'num_samples': len(generated_captions)}

        if HAVE_NLTK and generated_captions:
            try:
                # BLEU (corpus-level)
                refs = []
                hyps = []
                for img_id, gen in generated_captions.items():
                    if img_id not in self.test_captions:
                        continue
                    refs.append([r.split() for r in self.test_captions[img_id]])
                    hyps.append(gen.split())

                results.update({
                    'BLEU-1': corpus_bleu(refs, hyps, weights=(1, 0, 0, 0)),
                    'BLEU-2': corpus_bleu(refs, hyps, weights=(0.5, 0.5, 0, 0)),
                    'BLEU-3': corpus_bleu(refs, hyps, weights=(0.33, 0.33, 0.33, 0)),
                    'BLEU-4': corpus_bleu(refs, hyps, weights=(0.25, 0.25, 0.25, 0.25))
                })
            except Exception as e:
                print(f'Warning computing BLEU: {e}')

            try:
                # METEOR: average best-match per image
                meteor_scores = []
                for img_id, gen in generated_captions.items():
                    refs = self.test_captions.get(img_id, [])
                    if not refs:
                        continue
                    scs = [meteor_score([r.split()], gen.split()) for r in refs]
                    meteor_scores.append(max(scs) if scs else 0)
                results['METEOR'] = float(np.mean(meteor_scores)) if meteor_scores else 0.0
            except Exception as e:
                print(f'Warning computing METEOR: {e}')
        else:
            if not HAVE_NLTK:
                print('NLTK not available; skipping BLEU/METEOR')

        if HAVE_ROUGE and generated_captions:
            try:
                scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
                r1, r2, rL = [], [], []
                for img_id, gen in generated_captions.items():
                    refs = self.test_captions.get(img_id, [])
                    if not refs:
                        continue
                    per_ref_r1, per_ref_r2, per_ref_rL = [], [], []
                    for ref in refs:
                        sc = scorer.score(ref, gen)
                        per_ref_r1.append(sc['rouge1'].fmeasure)
                        per_ref_r2.append(sc['rouge2'].fmeasure)
                        per_ref_rL.append(sc['rougeL'].fmeasure)
                    r1.append(max(per_ref_r1) if per_ref_r1 else 0)
                    r2.append(max(per_ref_r2) if per_ref_r2 else 0)
                    rL.append(max(per_ref_rL) if per_ref_rL else 0)
                results.update({'ROUGE-1': float(np.mean(r1)) if r1 else 0.0,
                                'ROUGE-2': float(np.mean(r2)) if r2 else 0.0,
                                'ROUGE-L': float(np.mean(rL)) if rL else 0.0})
            except Exception as e:
                print(f'Warning computing ROUGE: {e}')
        else:
            if not HAVE_ROUGE:
                print('rouge_score not available; skipping ROUGE metrics')

        return results, generated_captions

    def visualize(self, generated_captions, image_dir='data/processed/images/', save_dir='results/predictions/', num_samples=5):
        os.makedirs(save_dir, exist_ok=True)

        if not generated_captions:
            print('No generated captions to visualize')
            return

        sample_ids = list(generated_captions.keys())[:num_samples]
        fig, axes = plt.subplots(len(sample_ids), 1, figsize=(12, 4*len(sample_ids)))

        if len(sample_ids) == 1:
            axes = [axes]

        for i, image_id in enumerate(sample_ids):
            image_path = os.path.join(image_dir, f"{image_id}.jpg")
            if not os.path.exists(image_path):
                continue
            img = Image.open(image_path)
            axes[i].imshow(img)
            axes[i].axis('off')
            ref = self.test_captions.get(image_id, [''])[0]
            gen = generated_captions.get(image_id, '')
            axes[i].set_title(f"Generated: {gen}\nReference: {ref}", fontsize=10, wrap=True)

        plt.tight_layout()
        out_path = os.path.join(save_dir, 'prediction_samples.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f'Saved visualization to {out_path}')

    def save_results(self, results, generated_captions, save_dir='results/smoke_eval'):
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'evaluation_results.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        with open(os.path.join(save_dir, 'generated_captions.json'), 'w', encoding='utf-8') as f:
            json.dump(generated_captions, f, indent=2)
        print(f'Saved results to {save_dir}')


def evaluate_smoke(model_path='models/smoke_models/football_caption_model.h5', processed_dir='data/processed', max_samples=50, out_dir='results/smoke_eval'):
    evaluator = FootballCaptionEvaluator(model_path=model_path, processed_dir=processed_dir)
    results, captions = evaluator.evaluate(max_samples=max_samples)
    evaluator.visualize(captions, num_samples=5, save_dir=os.path.join(out_dir, 'predictions'))
    evaluator.save_results(results, captions, save_dir=out_dir)
    return results, captions


if __name__ == '__main__':
    print('Running smoke evaluation')
    res, caps = evaluate_smoke()
    print('\nEvaluation summary:')
    for k, v in res.items():
        if isinstance(v, float):
            print(f'{k}: {v:.4f}')
        else:
            print(f'{k}: {v}')