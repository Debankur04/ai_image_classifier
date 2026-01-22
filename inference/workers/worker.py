# workflow
# 1- get data from redis
# 2- take manifest from bucket
# 3- start a result list
# 4- convert and process image using image_prep.py
# 5- if images = 2 make a batch and make prediction using prediction.py append in result
# 6- if image = 1 make a batch and make pred using prediction.py append in result
# 7- create pdf using result in pdf_creator.py
# 8- delete all images in the bucket then add the pdf in result and get the link
# 9- put the link in email.py

from redis import Redis
from workers import email, image_prep, pdf_creator, prediction
from supabase.supabase_init import supabase
import json

r = Redis(
    host="localhost",
    port=6379,
    db=0,
    decode_responses=True
)

QUEUE_NAME = "task_queue"

while True:
    task_data = r.blpop(QUEUE_NAME, timeout=0)
    if task_data:

        { "job_id": "b7f1c0b2-9e4f-4a9d-9d6f-8a2a9f71c123", 
        "user_id": "u_123456", 
        "user_email": "user@example.com", 
        "bucket": "user-uploads", 
        "input_prefix": "users/u_123456/jobs/b7f1c0b2/input/", 
        "manifest_path": "users/u_123456/jobs/b7f1c0b2/manifest.json", 
        "report_prefix": "users/u_123456/jobs/b7f1c0b2/report/" }

        bucket = task_data['bucket']
        input_prefix = task_data['input_prefix']
        manifest_path = task_data['manifest_path']
        results = []
        batch_images = []

        manifest = (
        supabase.storage
        .from_(bucket)
        .download(manifest_path))
        with open(manifest, 'r') as file:
            data = json.load(file)
        
        for relative_path in data['images']:
            image = image_prep(bucket = bucket, file_path = relative_path)
            batch_images.append(image)
            if len(batch_images) == 2:
                batch_result = prediction.predict_batch(batch_images)
                results.extend(batch_result)
                batch_images.clear()
        if batch_images == 1:
            batch_result = prediction.predict_batch(batch_images)
            results.extend(batch_result)
            batch_images.clear()
        pdf_creator.create_pdf_report(results= results)
        response = (
        supabase.storage
        .from_(bucket)
        .remove(input_prefix)
        )