import json
import traceback
import signal
import sys
from redis import Redis

from workers import email_worker, image_prep, pdf_creator, prediction
from supabase_client.supabase_init import supabase_admin
from supabase_client.storage_operations import (
    delete_images_create_report,
    create_signed_report_url
)
from supabase_client.db_operations import update_job_status


# =========================
# Graceful Shutdown
# =========================
def shutdown_handler(sig=None, frame=None):
    print("\n🛑 Worker shutdown requested. Exiting gracefully...")
    sys.exit(0)


signal.signal(signal.SIGINT, shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)


# =========================
# Worker Function
# =========================
def run_worker():
    r = Redis(
        host="localhost",
        port=6379,
        db=0,
        decode_responses=True
    )

    QUEUE_NAME = "task_queue"
    PROCESSING_QUEUE = "task_queue:PROGRESSED"

    MAX_IDLE_RETRIES = 5
    MAX_JOB_RETRIES = 5
    idle_retries = 0

    print("🚀 Worker started")
    print(f"📥 Waiting on Redis queue: {QUEUE_NAME}")
    print("🧠 Press Ctrl+C to stop safely\n")

    while True:
        try:
            print("⏳ Waiting for next job...")
            task_json = r.brpoplpush(
                QUEUE_NAME,
                PROCESSING_QUEUE,
                timeout=5
            )

            if not task_json:
                idle_retries += 1
                print(f"🫀 Worker alive, no jobs yet ({idle_retries}/{MAX_IDLE_RETRIES})")

                if idle_retries >= MAX_IDLE_RETRIES:
                    shutdown_handler()

                continue

            idle_retries = 0

            task = json.loads(task_json)
            job_id = task["job_id"]

            print(f"\n🔄 Picked up job: {job_id}")

            try:
                print("🟡 Updating job status → PROGRESSED")
                update_job_status(job_id=job_id, status="PROGRESSED")

                bucket = task["bucket"]
                input_prefix = task["input_prefix"]
                manifest_path = task["manifest_path"]
                report_prefix = task["report_prefix"]
                report_filename = task["report_filename"]

                results = []
                batch_images = []

                manifest_bytes = (
                    supabase_admin.storage
                    .from_(bucket)
                    .download(manifest_path)
                )

                manifest = json.loads(manifest_bytes.decode("utf-8"))

                for filename in manifest["images"]:
                    processed_img = image_prep.load_image(
                        bucket_name=bucket,
                        file_path=f"{input_prefix}{filename}"
                    )

                    batch_images.append(processed_img)

                    if len(batch_images) == 2:
                        results.extend(prediction.predict_batch(batch_images))
                        batch_images.clear()

                if batch_images:
                    results.extend(prediction.predict_batch(batch_images))

                pdf_creator.create_pdf_report(
                    results=results,
                    output_path=report_filename
                )

                report_path = delete_images_create_report(
                    bucket=bucket,
                    input_prefix=input_prefix,
                    report_prefix=report_prefix,
                    report_filename=report_filename
                )

                signed_url = create_signed_report_url(
                    bucket=bucket,
                    report_path=report_path
                )

                print("🟢 Updating job status → DONE")
                update_job_status(
                    job_id=job_id,
                    status="DONE",
                    report_path=report_path
                )

                r.delete(f"job_retry:{job_id}")

                email_worker.send_report_email(
                    user_email=task["user_email"],
                    user_id=task["user_id"],
                    report_link=signed_url
                )

                r.lrem(PROCESSING_QUEUE, 1, task_json)
                print(f"✅ Job {job_id} completed")

            except Exception:
                print(f"❌ Job {job_id} failed")
                print(traceback.format_exc())

                retry_key = f"job_retry:{job_id}"
                retries = r.incr(retry_key)

                if retries >= MAX_JOB_RETRIES:
                    update_job_status(job_id=job_id, status="FAILED")
                    r.lrem(PROCESSING_QUEUE, 1, task_json)
                    r.delete(retry_key)
                    print(f"⛔ Job {job_id} permanently failed")

                else:
                    r.lrem(PROCESSING_QUEUE, 1, task_json)
                    r.rpush(QUEUE_NAME, task_json)
                    print(f"🔁 Retrying job {job_id} ({retries}/{MAX_JOB_RETRIES})")

                continue

        except KeyboardInterrupt:
            shutdown_handler()


if __name__ == "__main__":
    run_worker()
