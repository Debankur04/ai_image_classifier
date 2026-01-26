from typing import Optional
from supabase_client.supabase_init import supabase_admin


# -----------------------------
# 1. Insert a new job
# -----------------------------
def insert_job(user_id: str, status: str = "QUEUED") -> str:
    try:
        response = (
            supabase_admin
            .table("jobs")
            .insert({
                "user_id": user_id,
                "status": status
            })
            .execute()
        )

        if not response.data:
            raise RuntimeError("Job insertion returned empty response")

        return response.data[0]["job_id"]

    except Exception as e:
        raise RuntimeError(f"[JOB INSERT FAILED] {str(e)}") from e


# ---------------------------------
# 2. Update job status
# ---------------------------------
def update_job_status(
    job_id: str,
    status: str,
    report_path: Optional[str] = None
) -> None:
    try:
        payload = {"status": status}

        if report_path:
            payload["report_path"] = report_path

        response = (
            supabase_admin
            .table("jobs")
            .update(payload)
            .eq("job_id", job_id)
            .execute()
        )

        if not response.data:
            raise RuntimeError("Job update affected 0 rows")

    except Exception as e:
        raise RuntimeError(f"[JOB UPDATE FAILED] {str(e)}") from e


# ---------------------------------
# 3. Delete a job
# ---------------------------------
def delete_job(job_id: str) -> None:
    try:
        response = (
            supabase_admin
            .table("jobs")
            .delete()
            .eq("job_id", job_id)
            .execute()
        )

        if not response.data:
            raise RuntimeError("Job delete affected 0 rows")

    except Exception as e:
        raise RuntimeError(f"[JOB DELETE FAILED] {str(e)}") from e


# ---------------------------------
# 4. Return all jobs for a user
# ---------------------------------
def returning_all_jobs(user_id: str):
    try:
        response = (
            supabase_admin
            .table("jobs")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .execute()
        )

        return response.data or []

    except Exception as e:
        raise RuntimeError(f"[FETCH JOBS FAILED] {str(e)}") from e
