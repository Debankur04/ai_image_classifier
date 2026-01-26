# 📘 Backend API Documentation

## 🔐 Authentication
- POST /auth/signup

Body
```
{
  "email": "user@example.com",
  "password": "StrongPass123!"
}
```

Response
```
{
  "user_id": "...",
  "access_token": "..."
}
```
- POST /auth/signin

Same as signup.

## 🧠 Jobs
- POST /jobs

Uploads images and creates job.
```
Headers

Authorization: Bearer <JWT>
```

Body
```
form-data

images[] (files)
```
Max 100 images

Max 5MB per image


Response
```
{
  "job_id": "...",
  "status": "QUEUED"
}
```
- GET /jobs

Returns all jobs for user.

- GET /jobs/{job_id}

Returns job details and status.

- GET /jobs/{job_id}/report

Downloads PDF report (auto-download).

- DELETE /jobs/{job_id}

Deletes job metadata.

## 🏥 Health
- GET /health
```
{
  "status": "ok"
}
```
