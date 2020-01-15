import os
import shutil
import zipfile
from urllib import parse

import kaggle
import requests
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from six.moves import urllib

from utils.dataset_format import unzip_nested_zip


def zip_file_name_from_url(data_url, data_dir):
    data_url_str = data_url.split('/')
    filename = parse.unquote(data_url_str[-1])
    zipfile_path = os.path.join(data_dir, filename)
    unzip_dir = os.path.join(data_dir, filename[0:-4])
    return zipfile_path, unzip_dir


def is_there_file(zipfile_path, unzip_dir):
    """if a file has existed"""
    if os.path.isfile(zipfile_path):
        # 如果存在zip文件就不用下载了，直接解压即可
        if os.path.isdir(unzip_dir):
            # 如果已经解压了就啥也不用做了
            return True
        unzip_nested_zip(zipfile_path, unzip_dir)
        return True


def download_one_zip(data_url, data_dir):
    """download one zip file from url as data_file"""
    zipfile_path, unzip_dir = zip_file_name_from_url(data_url, data_dir)
    if not is_there_file(zipfile_path, unzip_dir):
        if not os.path.isdir(unzip_dir):
            os.mkdir(unzip_dir)
        r = requests.get(data_url, stream=True)
        with open(zipfile_path, "wb") as py_file:
            for chunk in r.iter_content(chunk_size=1024):  # 1024 bytes
                if chunk:
                    py_file.write(chunk)
        unzip_nested_zip(zipfile_path, unzip_dir)


def download_small_zip(data_url, data_dir):
    """下载文件较小的zip文件并解压"""
    zipfile_path, unzip_dir = zip_file_name_from_url(data_url, data_dir)
    if not is_there_file(zipfile_path, unzip_dir):
        if not os.path.isdir(unzip_dir):
            os.mkdir(unzip_dir)
        zipfile_path, _ = urllib.request.urlretrieve(data_url, zipfile_path)
        unzip_nested_zip(zipfile_path, unzip_dir)


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
        print("Kaggle File is ready!")
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


def download_google_drive(client_secrets_file, google_drive_dir_name, download_dir):
    """从google drive下载文件，首先要下载好client_secrets.json文件，然后在jupyter里运行下
    https://github.com/OuyangWenyu/aqualord/blob/master/CloudStor/googledrive.ipynb
    里的代码，把文件夹下得到的mycreds.txt保存到client_secrets_file，这样就有一个本地的凭证了"""
    # 根据client_secrets.json授权
    gauth = GoogleAuth()
    gauth.LoadCredentialsFile(client_secrets_file)
    if gauth.credentials is None:
        # Authenticate if they're not there
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        # Refresh them if expired
        gauth.Refresh()
    else:
        # Initialize the saved creds
        gauth.Authorize()
    # Save the current credentials to a file
    gauth.SaveCredentialsFile(client_secrets_file)
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
        downloaded_files = os.listdir(download_dir)
        for file in file_list:
            print('title: %s, id: %s' % (file['title'], file['id']))
            if file['title'] in downloaded_files:
                continue
            file_dl = drive.CreateFile({'id': file['id']})
            print('Downloading file %s from Google Drive' % file_dl['title'])
            file_dl.GetContentFile(os.path.join(download_dir, file_dl['title']))
        print('Downloading files finished')
