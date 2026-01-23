from typing import List, Dict, Tuple
import os
import json

from supabase.supabase_init import supabase


# -----------------------------
# 1. Upload images + manifest
# -----------------------------
def upload_images_and_manifest( bucket: str, images: List[Tuple[str, str]], manifest: Dict, manifest_remote_path: str ) -> None:
    """
    Uploads images and manifest.json to a Supabase bucket.
    """
    try:
        # Upload images
        for local_path, remote_path in images:
            if not os.path.exists(local_path):
                raise FileNotFoundError(f"Image not found: {local_path}")

            with open(local_path, "rb") as f:
                supabase.storage.from_(bucket).upload(
                    path=remote_path,
                    file=f,
                    file_options={"content-type": "image/png"}
                )

        # Upload manifest.json
        manifest_bytes = json.dumps(manifest).encode("utf-8")
        supabase.storage.from_(bucket).upload(
            path=manifest_remote_path,
            file=manifest_bytes,
            file_options={"content-type": "application/json"}
        )

    except Exception as e:
        raise RuntimeError(f"[UPLOAD FAILED] {str(e)}") from e


# ---------------------------------
# 2. Download files as bytes
# ---------------------------------
# def download_files_as_bytes( bucket: str, paths: List[str] ) -> Dict[str, bytes]:
#     """
#     Downloads multiple files from Supabase bucket and returns {path: bytes}.
#     """
#     downloaded = {}

#     try:
#         for path in paths:
#             data = supabase.storage.from_(bucket).download(path)
#             if not data:
#                 raise ValueError(f"Empty download for {path}")
#             downloaded[path] = data

#         return downloaded

#     except Exception as e:
#         raise RuntimeError(f"[DOWNLOAD FAILED] {str(e)}") from e


# -------------------------------------------------
# 3.Upload the report, Delete all images, keep only the report
# -------------------------------------------------
def delete_images_create_report(
    bucket: str,
    input_prefix: str,
    report_prefix: str,
    report_filename: str
) -> str:
    """
    Uploads the final report, deletes all input images,
    and returns the report path.
    """
    try:
        # 🔹 Upload report FIRST (creates report prefix automatically)
        with open(report_filename, "rb") as pdf_file:
            supabase.storage.from_(bucket).upload(
                path=f"{report_prefix}/{report_filename}",
                file=pdf_file,
                file_options={"content-type": "application/pdf"}
            )

        # 🔹 Delete input images
        files = supabase.storage.from_(bucket).list(input_prefix)

        delete_targets = [
            f"{input_prefix}/{f['name']}"
            for f in files
        ]

        if delete_targets:
            supabase.storage.from_(bucket).remove(delete_targets)

        # ✅ return report path
        return f"{report_prefix}/{report_filename}"

    except Exception as e:
        raise RuntimeError(f"[DELETE FAILED] {str(e)}")




# ---------------------------------
# 4. Create signed URL (1 day)
# ---------------------------------
def create_signed_report_url( bucket: str, report_path: str, expiry_seconds: int = 86400 ) -> str:
    """
    Generates a signed URL for the report file.
    """
    try:
        response = supabase.storage.from_(bucket).create_signed_url(
            path=report_path,
            expires_in=expiry_seconds
        )

        signed_url = response.get("signedURL")
        if not signed_url:
            raise ValueError("Signed URL not returned")

        return signed_url

    except Exception as e:
        raise RuntimeError(f"[SIGNED URL FAILED] {str(e)}") from e
