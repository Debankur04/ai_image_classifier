from pydantic import BaseModel, EmailStr

class AuthPayload(BaseModel):
    email: EmailStr
    password: str


class JobCreateResponse(BaseModel):
    job_id: str
    status: str