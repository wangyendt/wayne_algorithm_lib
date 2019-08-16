# -*- coding: utf-8 -*-
# @Author  : Carry Chen
# @Email   : li.chen@newdegreetech.com, chenliworking@163.com
# @Software: PyCharm
# @FileName: project_common_use.py
# @Time    : 2019/8/15 21:50
# @Description: 项目常用的一些函数


def get_testLog_allData_pair_fn(file):
    """
    找到同名的testLog和allData的file name
    :param file: 完整路径的文件名，allData或者testLog的都可以
    :return:
    testLog和allData的file name
    """
    if 'allData.txt' in file:
        testLog_fn = file[:-10] + 'testLog.txt'
        allData_fn = file
    elif 'testLog.txt' in file:
        allData_fn = file[:-11] + 'allData.txt'
        testLog_fn = file
    if not (os.path.exists(testLog_fn) or os.path.exists(testLog_fn)):
        print(testLog_fn, '\n', testLog_fn)
        print('不存在指定文件')
        exit()
    return testLog_fn, allData_fn


def get_calibrateLog_allData_pair_fn(file):
    """
    找到同名的calibrateLog和allData的file name
    :param file: 完整路径的文件名，allData或者testLog的都可以
    :return:
    testLog和allData的file name
    """
    if 'allData.txt' in file:
        calibrateLog_fn = file[:-10] + 'calibrateLog.txt'
        allData_fn = file
    elif 'testLog.txt' in file:
        allData_fn = file[:-15] + 'allData.txt'
        testLog_fn = file
    if not (os.path.exists(calibrateLog_fn) or os.path.exists(calibrateLog_fn)):
        print(calibrateLog_fn, '\n', calibrateLog_fn)
        print('不存在指定文件')
        exit()
    return calibrateLog_fn, allData_fn


def read_log_data(file):
    """
    读取debug or alldata 的数组数据
    自动跳过前2行
    :param file: str  file path
    :return:
    data, np.array
    """
    with open(file, 'r', encoding='utf-8') as f:
        data = np.array(pd.read_csv(f, header=None, skiprows=2, delimiter='\t'))
    return data
