import os
import smtplib
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email import encoders

def send_email(subject, body, attachments=None):
    sender_email = 'rk3055089@gmail.com'
    recipient_email = 'rk3055089@gmail.com'
    app_password = 'pssz aqhi ooal weho'  # Use the App Password or API Key

    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = recipient_email
    message['Subject'] = subject
    message.attach(MIMEText(body, 'plain'))

    # Attach attachments if provided
    if attachments:
        for attachment in attachments:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(open(attachment, 'rb').read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename="{os.path.basename(attachment)}"')
            message.attach(part)

    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, app_password)
            server.sendmail(sender_email, recipient_email, message.as_string())
        print('Email notification sent successfully!')

    except smtplib.SMTPException as e:
        print(f'Error: Unable to send email. {str(e)}')