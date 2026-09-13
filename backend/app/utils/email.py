import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from app.config import settings

logger = logging.getLogger(__name__)

def send_otp_email(email: str, otp_code: str) -> bool:
    """Send an OTP code to a user's email address."""
    subject = "AI MADAC - Password Reset Verification Code"
    body = f"""
    Hello,

    You requested a password reset for your AI MADAC account.
    Your verification OTP code is:

    ===================================
                {otp_code}
    ===================================

    This code is valid for 10 minutes.
    If you did not request this reset, please ignore this email.

    Regards,
    AI MADAC Team
    """

    # Always log to stdout/logs for ease of development/testing
    logger.info("==============================================")
    logger.info(f"OTP FOR {email}: {otp_code}")
    logger.info("==============================================")
    print("==============================================")
    print(f"OTP FOR {email}: {otp_code}")
    print("==============================================")

    if not settings.SMTP_HOST or not settings.SMTP_USER or not settings.SMTP_PASSWORD:
        logger.warning("SMTP is not fully configured. OTP sent only to logs/terminal.")
        return False

    try:
        msg = MIMEMultipart()
        msg['From'] = settings.SMTP_FROM
        msg['To'] = email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=10)
        server.starttls()
        server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
        server.sendmail(settings.SMTP_FROM, email, msg.as_string())
        server.quit()
        logger.info(f"OTP successfully sent via email to {email}")
        return True
    except Exception as e:
        logger.error(f"Failed to send email OTP to {email}: {e}")
        return False
