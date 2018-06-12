# -*- coding: utf-8 -*-

from email.mime.text import MIMEText
import smtplib


def send_mail(body, subject='result summary', debug=False):
    to_addr = ''
    password = ''
    smtp_server = ''
    from_addr = ''

    msg = MIMEText(body, 'plain', 'utf-8')
    msg['subject'] = subject
    msg['from'] = from_addr
    msg['to'] = to_addr

    server = smtplib.SMTP_SSL(smtp_server, smtplib.SMTP_SSL_PORT)
    server.set_debuglevel(debug)
    server.login(from_addr, password)
    server.sendmail(from_addr, [to_addr], msg.as_string())
    server.quit()

    print 'email have been sent.'

# send_mail("the work has been finished.", debug=True)
