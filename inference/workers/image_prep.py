import tensorflow as tf
from inference.supabase_client.supabase_init import supabase_admin

IMG_SIZE = (224, 224)

def load_image(bucket_name: str, file_path: str):
    """
    bucket_name: Supabase storage bucket (e.g. 'avatars')
    file_path: path inside bucket (e.g. 'folder/avatar1.png')
    """

    # Download image bytes from Supabase
    image_bytes = (
        supabase_admin
        .storage
        .from_(bucket_name)
        .download(file_path)
    )

    # Decode image from bytes
    image = tf.image.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False
    )

    # Resize and normalize
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0

    return image

