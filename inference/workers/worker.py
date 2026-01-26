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
from inference.supabase_client.db_operations import update_job_status


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
    # ---------------- Redis Setup ----------------
    r = Redis(
        host="localhost",
        port=6379,
        db=0,
        decode_responses=True
    )

    QUEUE_NAME = "task_queue"
    PROCESSING_QUEUE = "task_queue:processing"

    MAX_IDLE_RETRIES = 5
    idle_retries = 0

    print("🚀 Worker started")
    print(f"📥 Waiting on Redis queue: {QUEUE_NAME}")
    print("🧠 Press Ctrl+C to stop safely\n")

    # =========================
    # Worker Loop
    # =========================
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
                    print("🛑 No jobs after multiple checks. Shutting down worker.")
                    shutdown_handler()

                continue

            # Reset idle counter on job pickup
            idle_retries = 0

            task = json.loads(task_json)
            job_id = task["job_id"]

            print("\n==============================")
            print(f"🔄 Picked up job: {job_id}")
            print(f"👤 User: {task['user_email']}")
            print("==============================")

            try:
                # ---------------- Update status → PROCESSING ----------------
                print("🟡 Updating job status → PROCESSING")
                update_job_status(job_id=job_id, status="PROCESSING")

                bucket = task["bucket"]
                input_prefix = task["input_prefix"]
                manifest_path = task["manifest_path"]
                report_prefix = task["report_prefix"]
                report_filename = task["report_filename"]

                results = []
                batch_images = []

                # ---------------- Download Manifest ----------------
                print("📖 Downloading manifest.json")
                manifest_bytes = (
                    supabase_admin.storage
                    .from_(bucket)
                    .download(manifest_path)
                )

                manifest = json.loads(manifest_bytes.decode("utf-8"))
                print(f"🖼️ Images to process: {len(manifest['images'])}")

                # ---------------- Process Images ----------------
                for idx, filename in enumerate(manifest["images"], start=1):
                    print(f"🖼️ Loading image {idx}: {filename}")

                    processed_img = image_prep.load_image(
                        bucket_name=bucket,
                        file_path=f"{input_prefix}{filename}"
                    )

                    batch_images.append(processed_img)

                    if len(batch_images) == 2:
                        print("🤖 Running prediction on batch of 2")
                        batch_result = prediction.predict_batch(batch_images)
                        results.extend(batch_result)
                        batch_images.clear()

                if batch_images:
                    print("🤖 Running prediction on final batch")
                    batch_result = prediction.predict_batch(batch_images)
                    results.extend(batch_result)
                    batch_images.clear()

                # ---------------- Create PDF ----------------
                print("📄 Creating PDF report")
                pdf_creator.create_pdf_report(
                    results=results,
                    output_path=report_filename
                )

                # ---------------- Upload Report & Cleanup ----------------
                print("☁️ Uploading report and deleting input images")
                report_path = delete_images_create_report(
                    bucket=bucket,
                    input_prefix=input_prefix,
                    report_prefix=report_prefix,
                    report_filename=report_filename
                )

                # ---------------- Signed URL ----------------
                print("🔐 Creating signed URL")
                signed_url = create_signed_report_url(
                    bucket=bucket,
                    report_path=report_path
                )

                # ---------------- Update status → DONE ----------------
                print("🟢 Updating job status → DONE")
                update_job_status(
                    job_id=job_id,
                    status="DONE",
                    report_path=report_path
                )

                # ---------------- Send Email ----------------
                print("📧 Sending report email")
                email_worker.send_report_email(
                    user_email=task["user_email"],
                    user_id=task["user_id"],
                    report_link=signed_url
                )

                # ---------------- ACK JOB ----------------
                r.lrem(PROCESSING_QUEUE, 1, task_json)
                print("🧹 Job removed from processing queue")
                print(f"✅ Job {job_id} completed successfully 🎉")

            except Exception:
                print(f"❌ Job {job_id} failed")
                print(traceback.format_exc())

                try:
                    print("🔴 Updating job status → FAILED")
                    update_job_status(job_id=job_id, status="FAILED")
                except Exception as db_err:
                    print("⚠️ Failed to update job status:", db_err)

                # ---------------- REQUEUE JOB ----------------
                r.lrem(PROCESSING_QUEUE, 1, task_json)
                r.rpush(QUEUE_NAME, task_json)

                print("🔁 Job requeued for retry (testing mode)")
                continue

        except KeyboardInterrupt:
            shutdown_handler()


# =========================
# Entrypoint
# =========================
if __name__ == "__main__":
    run_worker()
