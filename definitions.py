"""At beginning, copy kaggle.json and mycreds.txt to the /data directory manually"""
import os
import shutil
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
CONFIG_DIR = os.path.join(ROOT_DIR, "example/User")
CONFIG_FILE = os.path.join(CONFIG_DIR, 'config.ini')  # your own configuration file
if not os.path.isdir(CONFIG_DIR):
    os.mkdir(CONFIG_DIR)
if not os.path.isfile(CONFIG_FILE):
    shutil.copy(os.path.join(ROOT_DIR, 'data/config.ini'), CONFIG_FILE)

# kaggle file configuration
HOME_DIR = str(Path.home())
KAGGLE_DIR = os.path.join(HOME_DIR, ".kaggle")
KAGGLE_FILE = os.path.join(KAGGLE_DIR, "kaggle.json")
if not os.path.isdir(KAGGLE_DIR):
    os.mkdir(KAGGLE_DIR)
if not os.path.isfile(KAGGLE_FILE):
    shutil.move(os.path.join(ROOT_DIR, 'data/kaggle.json'), KAGGLE_DIR)
