"""各类日用工具"""
import json
import pickle
import smtplib
import ssl

import numpy as np


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


def serialize_json(my_dict, my_file):
    """将dict数据序列化到json文件
    :parameter
        my_dict:要存储的dict数据
        my_file:json文件具体地址
    """
    with open(my_file, 'w') as FP:
        json.dump(my_dict, FP, indent=4)


def unserialize_json(my_file):
    with open(my_file, 'r') as fp:
        my_object = json.load(fp)
    return my_object


def serialize_pickle(my_object, my_file):
    """python最经典序列化
    :parameter
         my_object:要序列化的python对象
         my_file:txt文件具体地址
    """
    f = open(my_file, 'wb')
    pickle.dump(my_object, f)
    f.close()


def unserialize_pickle(my_file):
    f = open(my_file, 'rb')
    my_object = pickle.load(f)
    f.close()
    return my_object


def serialize_numpy(my_array, my_file):
    """序列化numpy数组"""
    np.save(my_file, my_array)


def unserialize_numpy(my_file):
    y = np.load(my_file)
    return y
