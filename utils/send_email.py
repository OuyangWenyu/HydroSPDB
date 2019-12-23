import smtplib, ssl


def send_email(subject, text, receiver='hust2014owen@gmail.com'):
    """用于训练结束时发邮件提醒"""
    sender = 'hydro.wyouyang@gmail.com'
    password = 'D4VEFya3UQxGR3z'
    context = ssl.create_default_context()
    msg = 'Subject: {}\n\n{}'.format(subject, text)
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender, password)
        server.sendmail(
            from_addr=sender, to_addrs=receiver, msg=msg)
