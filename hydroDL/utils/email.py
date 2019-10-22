import smtplib, ssl


# def sendEmail(subject, text, receiver='geofkwai@gmail.com'):
#     sender = 'fkwai.public@gmail.com'
#     password = 'fkwai0323'
#     context = ssl.create_default_context()
#     msg = 'Subject: {}\n\n{}'.format(subject, text)
#     with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
#         server.login(sender, password)
#         server.sendmail(
#             from_addr=sender, to_addrs=receiver, msg=msg)

def sendEmail(subject, text, receiver='hust2014owen@gmail.com'):
    sender = 'hydro.wyouyang@gmail.com'
    password = 'Psu!2019Owen'
    context = ssl.create_default_context()
    msg = 'Subject: {}\n\n{}'.format(subject, text)
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender, password)
        server.sendmail(
            from_addr=sender, to_addrs=receiver, msg=msg)
