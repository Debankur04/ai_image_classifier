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
def download_files_as_bytes( bucket: str, paths: List[str] ) -> Dict[str, bytes]:
    """
    Downloads multiple files from Supabase bucket and returns {path: bytes}.
    """
    downloaded = {}

    try:
        for path in paths:
            data = supabase.storage.from_(bucket).download(path)
            if not data:
                raise ValueError(f"Empty download for {path}")
            downloaded[path] = data

        return downloaded

    except Exception as e:
        raise RuntimeError(f"[DOWNLOAD FAILED] {str(e)}") from e


# -------------------------------------------------
# 3. Delete all images, keep only the report
# -------------------------------------------------
def delete_images_keep_report( bucket: str, folder_path: str, report_filename: str ) -> None:
    """
    Deletes all files in a folder except the final report.
    """
    try:
        files = supabase.storage.from_(bucket).list(folder_path)

        delete_targets = [
            f"{folder_path}/{f['name']}"
            for f in files
            if f["name"] != report_filename
        ]

        if delete_targets:
            supabase.storage.from_(bucket).remove(delete_targets)

    except Exception as e:
        raise RuntimeError(f"[DELETE FAILED] {str(e)}") from e


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
