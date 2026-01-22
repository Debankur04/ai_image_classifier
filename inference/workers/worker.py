import json
import time
from redis import Redis
from workers import email, image_prep, pdf_creator, prediction
from supabase.supabase_init import supabase

# ---------------- Redis ----------------
r = Redis(
    host="localhost",
    port=6379,
    db=0,
    decode_responses=True
)

QUEUE_NAME = "task_queue"

print("Worker started...")

while True:
    task_data = r.blpop(QUEUE_NAME, timeout=0)

    if not task_data:
        continue

    _, task_json = task_data
    task = json.loads(task_json)

    print(f"Processing job {task['job_id']}")

    bucket = task["bucket"]
    input_prefix = task["input_prefix"]
    manifest_path = task["manifest_path"]
    report_prefix = task["report_prefix"]

    results = []
    batch_images = []

    # -------- Load Manifest --------
    manifest_bytes = (
        supabase.storage
        .from_(bucket)
        .download(manifest_path)
    )

    manifest = json.loads(manifest_bytes.decode("utf-8"))

    # -------- Process Images --------
    for relative_path in manifest["images"]:
        processed_img = image_prep.load_image(
            bucket=bucket,
            file_path=relative_path
        )

        batch_images.append(processed_img)

        if len(batch_images) == 2:
            batch_result = prediction.predict_batch(batch_images)
            results.extend(batch_result)
            batch_images.clear()

    # Remaining single image
    if len(batch_images) == 1:
        batch_result = prediction.predict_batch(batch_images)
        results.extend(batch_result)
        batch_images.clear()

    # -------- Create PDF --------
    pdf_path = pdf_creator.create_pdf_report(
        results=results,
        output_path="report.pdf"
    )

    # -------- Upload PDF --------
    with open(pdf_path, "rb") as f:
        supabase.storage.from_(bucket).upload(
            path=f"{report_prefix}/report.pdf",
            file=f,
            file_options={"content-type": "application/pdf"}
        )

    # -------- Generate Signed URL --------
    signed_url = (
        supabase.storage
        .from_(bucket)
        .create_signed_url(
            path=f"{report_prefix}/report.pdf",
            expires_in=3600
        )
    )["signedURL"]

    # -------- Cleanup Input Files --------
    files = (
        supabase.storage
        .from_(bucket)
        .list(input_prefix)
    )

    file_paths = [f"{input_prefix}/{f['name']}" for f in files]

    if file_paths:
        supabase.storage.from_(bucket).remove(file_paths)

    # -------- Send Email --------
    email.send_report_email(
        user_email=task["user_email"],
        user_id=task["user_id"],
        report_link=signed_url
    )

    print(f"Job {task['job_id']} completed ✅")
