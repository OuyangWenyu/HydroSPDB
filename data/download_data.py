import os
import shutil
import zipfile
from urllib import parse

import kaggle
import requests
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from six.moves import urllib

from utils.dataset_format import unzip_file


def download_small_zip(data_url, data_dir):
    """下载文件较小的zip文件并解压"""
    data_url_str = data_url.split('/')
    filename = parse.unquote(data_url_str[-1])
    zipfile_path = os.path.join(data_dir, filename)
    unzip_dir = os.path.join(data_dir, filename[0:-4])
    if os.path.isfile(zipfile_path):
        # 如果存在zip文件就不用下载了，直接解压即可
        if os.path.isdir(unzip_dir):
            # 如果已经解压了就啥也不用做了
            return
        unzip_file(zipfile_path, unzip_dir)
        return
    if not os.path.isdir(unzip_dir):
        os.mkdir(unzip_dir)
    zipfile_path, _ = urllib.request.urlretrieve(data_url, zipfile_path)

    with zipfile.ZipFile(zipfile_path, 'r') as _zip:
        _zip.extractall(unzip_dir)


def download_small_file(data_url, temp_file):
    """根据url下载数据到temp_file中"""
    r = requests.get(data_url)
    with open(temp_file, 'w') as f:
        f.write(r.text)


def download_kaggle_file(kaggle_json, name_of_dataset, path_download, file_download):
    """下载kaggle上的数据，首先要下载好kaggle_json文件。 从kaggle上下载好数据之后，将其解压
    :parameter
        kaggle_json：认证文件
        name_of_dataset：kaggle上的数据文件名称
        path_download：下载数据到本地的文件夹
    """
    # 如果已经有了shp文件，就不需要再下载了
    if os.path.isfile(file_download):
        print("File is ready!")
        return
    home_dir = os.environ['HOME']
    kaggle_dir = os.path.join(home_dir, '.kaggle')
    if not os.path.isdir(kaggle_dir):
        os.mkdir(os.path.join(home_dir, '.kaggle'))

    dst = os.path.join(kaggle_dir, 'kaggle.json')
    if not os.path.isfile(dst):
        print("copying file...")
        if not os.path.isfile(kaggle_json):
            print("Please downloading your kaggle.json to this directory: ", kaggle_dir)
        shutil.copy(kaggle_json, dst)

    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(name_of_dataset, path=path_download)

    # 下载数据之后，解压在指定文件夹，起名为指定名称
    dataset_zip = name_of_dataset.split("/")[-1] + ".zip"
    dataset_zip = os.path.join(path_download, dataset_zip)
    with zipfile.ZipFile(dataset_zip, 'r') as zip_temp:
        zip_temp.extractall(path_download)


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
