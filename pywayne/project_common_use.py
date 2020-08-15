# -*- coding: utf-8 -*-
# @Author  : Carry Chen
# @Email   : li.chen@newdegreetech.com, chenliworking@163.com
# @Software: PyCharm
# @FileName: project_common_use.py
# @Time    : 2019/8/15 21:50
# @Description: 项目常用的一些函数

import numpy as np
import pandas as pd


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


def read_testLog(file, chs=1):
    """
    读取testLog文件
    :param file:
    :param chs:
    :return:
    """
    with open(file, 'r', encoding='utf-8') as f:
        # 1. 版本
        print(f.readline()[:-1])
        txtline = f.readline()
        print(f.readline()[:-1])
        # 2. 芯片温度
        while 'chip_temperature' not in txtline:
            txtline = f.readline()
            if txtline == '':
                print('文件错误,没有找到chip_temperature')
                exit()
        print(txtline)
        chip_t = float(txtline.split('\t')[1])
        # 3. coef
        while 'Coef' not in txtline:
            txtline = f.readline()
            if txtline == '':
                print('文件错误，没有找到Coef')
                exit()
        coef = list(map(float, txtline.split('\t')[1].split(',')[:-1]))
        # 4. Coef_temperature
        txtline = f.readline()
        coef_t = list(map(float, txtline.split('\t')[1].split(',')[:-1]))
        # 5.各个通道
        ADC, force = [], []
        for ch in range(chs):
            keyword = 'ch' + str(ch + 1) + '_0_avg'
            while keyword not in txtline:
                txtline = f.readline()
                if txtline == '':
                    print('文件错误，没有找到' + keyword)
                    exit()
            # ADC
            tmp0 = float(txtline.split('\t')[1])
            f.readline()
            txtline = f.readline()
            tmp1 = float(txtline.split('\t')[1])
            ADC.append(tmp1 - tmp0)
            # force
            txtline = f.readline()
            tmp = float(txtline.split('\t')[1])
            force.append(tmp)
        force, ADC, coef = np.array(force).astype(np.int), np.array(ADC).astype(np.int), np.array(coef).astype(np.int)
    return force, ADC, coef, coef_t, chip_t


def read_calibrateLog(file, chs=7):
    def read_adc_coef():
        adc_ch = float(txtline.split('\t')[1])
        txtline = f.readline()
        coef_ch = float(txtline.split('\t')[1])
        return adc_ch, coef_ch

    with open(file, 'r', encoding='utf-8') as f:
        # 1. 打印APK和固件版本
        print(f.readline()[:-1])
        txtline = f.readline()
        print(f.readline()[:-1])
        # 2. 打印和读取芯片温度
        while 'chip_temperature' not in txtline:
            txtline = f.readline()
            if txtline == '':
                print('文件错误')
                exit()
        print(txtline)
        chip_t = float(txtline.split('\t')[1])
        # 3. 各个通道
        ADC, coef = [], []
        kwd_old = 'impossible word'
        for ch in range(chs):
            keyword = 'ch' + str(ch + 1) + '__diff'
            while keyword not in txtline:
                txtline = f.readline()
                if kwd_old in txtline:
                    adc_ch, coef_ch = read_adc_coef()
                    ADC[-1], coef[-1] = adc_ch, coef_ch
                if txtline == '':
                    print('文件错误，没有找到' + keyword)
                    exit()
            kwd_old = keyword
            adc_ch, coef_ch = read_adc_coef()
            ADC.append(adc_ch)
            coef.append(coef_ch)
    return ADC, coef, chip_t


def read_log_data(file: str, skprows: int = 2, delimiter: str = '\t') -> np.array:
    """
    读取debug or alldata 的数组数据
    自动跳过前2行
    :param file: str  file path
    :return:
    data, np.array
    """
    with open(file, 'r', encoding='utf-8') as f:
        data = np.array(pd.read_csv(f, header=None, skiprows=skprows, delimiter=delimiter))
    return data
