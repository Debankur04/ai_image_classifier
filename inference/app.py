from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Depends,
)
from fastapi.responses import RedirectResponse
from typing import List
from datetime import datetime
import json

from redis import Redis
from fastapi.middleware.cors import CORSMiddleware
from supabase_client.supabase_init import supabase_public
from supabase_client.auth import signup, signin, signout
from supabase_client.storage_operations import upload_images_and_manifest, create_signed_report_url
from supabase_client.db_operations import (
    insert_job,
    delete_job,
    returning_all_jobs
)
from schema import AuthPayload, JobCreateResponse
from auth_dependency import get_current_user
from workers.worker import run_worker
# ---------------- App ----------------
app = FastAPI(
    title="AI Image Detection API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Redis ----------------
redis = Redis(
    host="localhost",
    port=6379,
    db=0,
    decode_responses=True
)
QUEUE_NAME = "task_queue"

# ---------------- Constants ----------------
BUCKET = "user-uploads"
MAX_IMAGES = 100
MAX_IMAGE_SIZE_MB = 5


# ============================================================
# Health
# ============================================================

@app.get("/health")
def health():
        return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat()
        }
    

# ============================================================
# Auth APIs
# ============================================================

@app.post("/auth/signup")
def signup_api(payload: AuthPayload):
    return signup(
         payload.email,
         payload.password
    )


@app.post("/auth/signin")
def signin_api(payload: AuthPayload):
    return signin(
        payload.email,
        payload.password
    )


@app.post("/auth/signout")
def signout_api():
    return signout()

# ============================================================
# Create Job
# ============================================================

@app.post("/jobs", response_model=JobCreateResponse)
async def create_job(
    images: List[UploadFile] = File(...),
    user=Depends(get_current_user)
):
    if not images:
        raise HTTPException(status_code=400, detail="No images provided")

    if len(images) > MAX_IMAGES:
        raise HTTPException(status_code=400, detail="Too many images")

    image_payload = []

    for img in images:
        content = await img.read()
        size_mb = len(content) / (1024 * 1024)

        if size_mb > MAX_IMAGE_SIZE_MB:
            raise HTTPException(
                status_code=400,
                detail=f"{img.filename} exceeds {MAX_IMAGE_SIZE_MB}MB"
            )

        image_payload.append((img.filename, content))

    # ---------------- Create Job ----------------
    job_id = insert_job(user_id=user["user_id"], status="QUEUED")

    base_path = f"users/{user['user_id']}/jobs/{job_id}"
    input_prefix = f"{base_path}/input"
    manifest_path = f"{base_path}/manifest.json"
    report_prefix = f"{base_path}/report"

    manifest = {
        "job_id": job_id,
        "user_id": user["user_id"],
        "images": [name for name, _ in image_payload],
        "total_images": len(image_payload),
        "created_at": datetime.utcnow().isoformat()
    }

    upload_images_and_manifest(
        bucket=BUCKET,
        images=image_payload,
        manifest=manifest,
        manifest_remote_path=manifest_path,
        input_prefix=input_prefix
    )

    # ---------------- Push to Redis ----------------
    redis_payload = {
        "job_id": job_id,
        "user_id": user["user_id"],
        "user_email": user["email"],
        "bucket": BUCKET,
        "input_prefix": f"{input_prefix}/",
        "manifest_path": manifest_path,
        "report_prefix": f"{report_prefix}/",
        "report_filename": "ai_image_report.pdf",
        "created_at": datetime.utcnow().isoformat()
    }

    redis.rpush(QUEUE_NAME, json.dumps(redis_payload))

    return JobCreateResponse(job_id=job_id, status="QUEUED")

# ============================================================
# List Jobs
# ============================================================

@app.get("/jobs")
def list_jobs(user=Depends(get_current_user)):
    return returning_all_jobs(user["user_id"])

# ============================================================
# Job Status
# ============================================================

@app.get("/jobs/{job_id}")
def get_job(job_id: str, user=Depends(get_current_user)):
    job = (
        supabase_public
        .table("jobs")
        .select("*")
        .eq("job_id", job_id)
        .eq("user_id", user["user_id"])
        .single()
        .execute()
    )

    if not job.data:
        raise HTTPException(status_code=404, detail="Job not found")

    return job.data

# ============================================================
# Download Report (auto-download)
# ============================================================

@app.get("/jobs/{job_id}/report")
def download_report(job_id: str, user=Depends(get_current_user)):
    job = (
        supabase_public
        .table("jobs")
        .select("report_path")
        .eq("job_id", job_id)
        .eq("user_id", user["user_id"])
        .single()
        .execute()
    )

    if not job.data or not job.data["report_path"]:
        raise HTTPException(status_code=404, detail="Report not ready")

    signed_url = create_signed_report_url(
        bucket=BUCKET,
        report_path=job.data["report_path"]
    )

    return RedirectResponse(url=signed_url)

# ============================================================
# Delete Job
# ============================================================

@app.delete("/jobs/{job_id}")
def delete_job_api(job_id: str, user=Depends(get_current_user)):
    delete_job(job_id)
    return {
        "job_id": job_id,
        "status": "deleted"
    }

@app.post("/internal/run-worker")
def run_worker_once():
    """
    Manually triggers the worker once.
    Blocking call. For testing only.
    """
    run_worker()

    return {
        "status": "finished",
        "message": "Worker run completed"
    }