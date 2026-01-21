import tensorflow as tf
from pathlib import Path
import keras

INFERENCE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = INFERENCE_DIR / "models" / "ai_vs_real_cnn.keras"

model = keras.models.load_model(MODEL_PATH)

# print('model loaded')
# model.summary()
# print('done')


def ensure_valid_batch(images):
    if not (1 <= len(images) <= 2):
        raise ValueError("Batch must contain 1 or 2 images.")
    

def predict_batch(images, threshold=0.5):
    """
    images: list of 1 or 2 tensors, each (224, 224, 3)
    """

    ensure_valid_batch(images)

    
    batch = tf.stack(images, axis=0)

    probs = model.predict(batch)

    results = []
    for i, prob in enumerate(probs):
        p = float(prob[0])

        label = "AI Generated" if p >= threshold else "Real"
        confidence = p if p >= threshold else 1 - p

        results.append({
            "image_tensor": images[i],
            "prediction": label,
            "confidence": round(confidence * 100, 2)
        })

    return results


