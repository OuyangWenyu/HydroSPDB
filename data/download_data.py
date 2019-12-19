import os
import shutil
import zipfile
from urllib import parse

import kaggle
import requests
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from six.moves import urllib


def download_small_zip(data_url, data_dir):
    """下载文件较小的zip文件并解压"""
    data_url_str = data_url.split('/')
    filename = parse.unquote(data_url_str[-1])
    filepath = os.path.join(data_dir, filename)
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    filepath, _ = urllib.request.urlretrieve(data_url, filepath)

    with zipfile.ZipFile(filepath, 'r') as _zip:
        _zip.extractall(data_dir)


def download_small_file(data_url, temp_file):
    """根据url下载数据到temp_file中"""
    r = requests.get(data_url)
    with open(temp_file, 'w') as f:
        f.write(r.text)


def download_kaggle_file(kaggle_json, name_of_dataset, path_download):
    """下载kaggle上的数据，首先要下载好kaggle_json文件"""
    home_dir = os.environ['HOME']
    kaggle_dir = os.path.join(home_dir, '.kaggle')
    print(home_dir)
    print(kaggle_dir)
    print(os.path.isdir(kaggle_dir))
    if not os.path.isdir(kaggle_dir):
        os.mkdir(os.path.join(home_dir, '.kaggle'))
    print(os.path.isdir(kaggle_dir))

    kaggle_dir = os.path.join(home_dir, '.kaggle')
    dst = os.path.join(kaggle_dir, 'kaggle.json')
    if not os.path.isfile(dst):
        print("copying file...")
        shutil.copy(kaggle_json, dst)

    kaggle.api.authenticate()

    kaggle.api.dataset_download_files(name_of_dataset, path=path_download)


def download_google_drive(google_drive_dir_name, download_dir):
    """从google drive下载文件，首先要下载好client_secrets.json文件"""
    # 根据client_secrets.json授权
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)
    # 先从google drive根目录判断是否有dir_name这一文件夹，没有的话就直接报错即可。
    dir_id = None
    file_list_root = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
    for file_temp in file_list_root:
        print('title: %s, id: %s' % (file_temp['title'], file_temp['id']))
        if file_temp['title'] == google_drive_dir_name:
            dir_id = str(file_temp['id'])
            #  列出该文件夹下的文件
            print('该文件夹的id是：', dir_id)
    if dir_id is None:
        print("No data....")
    else:
        # Auto-iterate through all files that matches this query
        file_list = drive.ListFile({'q': "'" + dir_id + "' in parents and trashed=false"}).GetList()
        for file in file_list:
            print('title: %s, id: %s' % (file['title'], file['id']))
            file_dl = drive.CreateFile({'id': file['id']})
            print('Downloading file %s from Google Drive' % file_dl['title'])
            file_dl.GetContentFile(os.path.join(download_dir, file_dl['title']))
        print('Downloading files finished')