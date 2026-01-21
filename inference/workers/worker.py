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
        bucket = task_data['bucket']
        input_prefix = task_data['input_prefix']
        manifest_path = task_data['manifest_path']