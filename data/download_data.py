import os
import urllib
from urllib import parse

import requests
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

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
        if os.path.isdir(unzip_dir):
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
        unzip_nested_zip(zipfile_path, unzip_dir), download_small_file


def download_small_zip(data_url, data_dir):
    """download zip file and unzip"""
    zipfile_path, unzip_dir = zip_file_name_from_url(data_url, data_dir)
    if not is_there_file(zipfile_path, unzip_dir):
        if not os.path.isdir(unzip_dir):
            os.mkdir(unzip_dir)
        zipfile_path, _ = urllib.request.urlretrieve(data_url, zipfile_path)
        unzip_nested_zip(zipfile_path, unzip_dir)


def download_small_file(data_url, temp_file):
    """download data from url to the temp_file"""
    r = requests.get(data_url)
    with open(temp_file, 'w') as f:
        f.write(r.text)


def download_excel(data_url, temp_file):
    """download a excel file according to url"""
    if not os.path.isfile(temp_file):
        urllib.request.urlretrieve(data_url, temp_file)


def download_a_file_from_google_drive(drive, dir_id, download_dir):
    file_list = drive.ListFile({'q': "'" + dir_id + "' in parents and trashed=false"}).GetList()
    for file in file_list:
        print('title: %s, id: %s' % (file['title'], file['id']))
        file_dl = drive.CreateFile({'id': file['id']})
        print('mimetype is %s' % file_dl['mimeType'])
        if file_dl['mimeType'] == 'application/vnd.google-apps.folder':
            download_dir_sub = os.path.join(download_dir, file_dl['title'])
            if not os.path.isdir(download_dir_sub):
                os.makedirs(download_dir_sub)
            download_a_file_from_google_drive(drive, file_dl['id'], download_dir_sub)
        else:
            # download
            temp_file = os.path.join(download_dir, file_dl['title'])
            if os.path.isfile(temp_file):
                print('file has been downloaded')
                continue
            file_dl.GetContentFile(os.path.join(download_dir, file_dl['title']))
            print('Downloading file finished')


def download_google_drive(client_secrets_file, google_drive_dir_name, download_dir):
    """mycreds.txt was got by the method in
    https://github.com/OuyangWenyu/aqualord/blob/master/CloudStor/googledrive.ipynb """
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

    dir_id = None
    file_list_root = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
    for file_temp in file_list_root:
        print('title: %s, id: %s' % (file_temp['title'], file_temp['id']))
        if file_temp['title'] == google_drive_dir_name:
            dir_id = str(file_temp['id'])
            print('the id of the dir is：', dir_id)
    if dir_id is None:
        print("No data....")
    else:
        download_a_file_from_google_drive(drive, dir_id, download_dir)
