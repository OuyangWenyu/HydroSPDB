import os
import shutil

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
CONFIG_DIR = os.path.join(ROOT_DIR, "example/User")
CONFIG_FILE = os.path.join(CONFIG_DIR, 'config.ini')  # your own configuration file
if not os.path.isdir(CONFIG_DIR):
    os.mkdir(CONFIG_DIR)
if not os.path.isfile(CONFIG_FILE):
    shutil.copy(os.path.join(ROOT_DIR, 'data/config.ini'), CONFIG_FILE)
