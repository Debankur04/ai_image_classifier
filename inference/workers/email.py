import os
import resend


resend.api_key = os.getenv("RESEND_API_KEY")


def send_report_email(user_email: str, user_id: str, report_link: str):
    """
    Sends report download link to the user via Resend.
    Returns 200 on success.
    """

    if not resend.api_key:
        raise RuntimeError("RESEND_API_KEY not configured")

    try:
        resend.Emails.send({
            "from": "AI Image Detection <no-reply@yourdomain.com>",
            "to": [user_email],
            "subject": "Your AI Image Detection Report is Ready",
            "html": f"""
                <p>Hi <b>{user_id}</b>,</p>

                <p>Your AI Image Detection report is ready.</p>

                <p>
                    <a href="{report_link}" target="_blank">
                        👉 Download your report
                    </a>
                </p>

                <p>
                    <i>
                    Disclaimer: This report was generated using an AI-based system
                    and may contain inaccuracies.
                    </i>
                </p>

                <p>Thanks,<br/>AI Image Detection Team</p>
            """
        })

        return 200

    except Exception as e:
        print(f"Resend email failed: {e}")
        raise
