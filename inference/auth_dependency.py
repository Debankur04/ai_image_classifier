from fastapi import (
    HTTPException,
    Header,
)
from supabase_client.supabase_init import supabase_public


def get_current_user(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization header")

    token = authorization.split(" ")[1]
    user = supabase_public.auth.get_user(token)

    if not user or not user.user:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    return {
        "user_id": user.user.id,
        "email": user.user.email
    }
