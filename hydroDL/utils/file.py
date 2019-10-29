"""文件格式转换等"""

import codecs

import chardet


def check_file_code(file_name):
    with open(file_name, 'r') as f:
        data = f.read()
        print(chardet.detect(data))


if __name__ == '__main__':
    check_file_code("C:\\Users\\hust2\\Desktop\\conterm_bas_classif.txt")
    print("It's done")
