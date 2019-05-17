import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

from ndtpy.tools import list_all_files
from ndtpy.tools import LoadData
from ndtpy.fw.utils import FindLocalExtremeValue

from ndtpy.fw.baseline_diff_v1 import BaseLine, leave_type_dict
# from .baseline_diff_v1 import BaseLine, leave_type_dict

plt.rcParams['font.family'] = 'FangSong'
plt.rcParams['axes.unicode_minus'] = False
ORDER1 = 16
CH_NUMS = 6


def __statistics(rootdir=r'..\\dataset_all', fid_bg=0, fid_ed=9999):
    def preparedir(dir0):
        if os.path.exists(dir0):
            shutil.rmtree(dir0)
        os.mkdir(dir0)

    # 准备文件夹
    preparedir('.\\0延迟释放数据集')
    preparedir('.\\0断触数据集')
    preparedir('.\\0不释放数据集')

    files = list_all_files(rootdir, ['.txt'], [])
    total_case, slow_release_case, early_stop_case, not_release_case = \
        len(files[fid_bg:fid_ed]), 0, 0, 0
    flev = FindLocalExtremeValue(local_window=30)
    data_loader = LoadData(channel_num=6, begin_ind_dic={
        'rawdata': 0,
        'baseline': 6,
        'forcesig': 12,
        'forceflag': 18,
        'humanflag': 19
    }, data_type='slice')
    for fid, path in enumerate(files[fid_bg:fid_ed]):
        print(fid, total_case)
        # load 文件
        data, rawdata, baseline_fw, forceflag_fw, forcesig_fw, forceflag_true, temperature_err_flag_fw, sig_err_flag \
            = data_loader.load_data(path)  # 导入切片数据
        # 起始点设置
        _, local_gap, _, _, _ = flev.run(rawdata[:101, 0])
        _, local_gap1, _, _, _ = flev.run(rawdata[:101, 2])
        local_gap = min(local_gap, local_gap1)
        start = int(101 - local_gap) - 10
        #
        bsl = BaseLine(rawdata)
        statistic_type = bsl.run_example(start_frame=start, forceflag_true=forceflag_true,
                                         statistic_flag=True)
        # 统计
        if statistic_type == 1:  # 延迟释放
            slow_release_case += 1
            shutil.copy(path, '.\\0延迟释放数据集\\' + path.split('\\')[-1])
        elif statistic_type == 3:  # 断触
            early_stop_case += 1
            shutil.copy(path, '.\\0断触数据集\\' + path.split('\\')[-1])
        elif statistic_type == 2:
            not_release_case += 1
            shutil.copy(path, '.\\0不释放数据集\\' + path.split('\\')[-1])
    # 打印统计结果
    print('一共有%d cases,%d个延迟释放(%0.2f)%%，%d个不释放(%0.2f)%%,%d个断触(%0.2f)%%' %
          (total_case,
           slow_release_case, slow_release_case / (total_case + 1e-8) * 100,
           not_release_case, not_release_case / (total_case + 1e-8) * 100,
           early_stop_case, early_stop_case / (total_case + 1e-8) * 100))
    return None


def __analysis_supervised_data(rootdir, compare_baseline=False):
    files = list_all_files(rootdir, ['.txt'], [])
    flev = FindLocalExtremeValue(local_window=30)
    data_loader = LoadData(channel_num=6, begin_ind_dic={
        'rawdata': 0,
        'baseline': 6,
        'forcesig': 12,
        'forceflag': 18,
        'humanflag': 19
    }, data_type='slice')
    for fid, path in enumerate(files[0:]):
        print(fid, path)
        data, rawdata, baseline_fw, forceflag_fw, forcesig_fw, forceflag_true, temperature_err_flag_fw, sig_err_flag \
            = data_loader.load_data(path)  # 导入切片数据
        _, local_gap, _, _, _ = flev.run(rawdata[:101, 0])
        _, local_gap1, _, _, _ = flev.run(rawdata[:101, 2])
        local_gap = min(local_gap, local_gap1)
        start = int(101 - local_gap) - 10
        print('start_frame', start)
        suptitle = path.split('\\')[-1][:-4]
        bsl = BaseLine(rawdata)
        if compare_baseline:
            bsl.run_example(start_frame=start, plot_out_flag=True, suptitle=suptitle,
                            forceflag_true=forceflag_true, baseline_fw=baseline_fw)
        else:
            bsl.run_example(start_frame=start, plot_out_flag=True, suptitle=suptitle,
                            forceflag_true=forceflag_true)
    return None


def __analysis_unsupervised_data(rootdir, compare_baseline=True, compare_forceflag=True, compare_stable_state=False):
    files = list_all_files(rootdir, ['.txt'], [])
    data_loader = LoadData(channel_num=6, begin_ind_dic={
        'rawdata': 28,
        'baseline': 34,
        'forcesig': 40,
        'forceflag': 1,
        'stableflag': 4,
        'temperature': 8,
        'fwversion': 26
    }, data_type='normal')
    for fid, path in enumerate(files[0:]):
        print('haha', fid, path)
        data, rawdata, baseline_fw, forceflag_fw, forcesig_fw, temperature_err_flag_fw, sig_err_flag, stable_state_fw \
            = data_loader.load_data(path)  # 导入完整文件
        start = 100
        suptitle = path.split('\\')[-1][:-4]
        bsl_fw = baseline_fw if compare_baseline else None
        ss_fw = stable_state_fw if compare_stable_state else None
        ff_fw = forceflag_fw if compare_forceflag else None
        bsl = BaseLine(rawdata)
        print(start)
        bsl.run_example(start_frame=start, plot_out_flag=True, suptitle=suptitle, baseline_fw=bsl_fw,
                        forceflag_fw=ff_fw, stable_state_fw=ss_fw)
    return


def test_baseline():
    # 选定文件夹
    rootdir1 = r'.\\0不释放数据集'  # 数据总文件夹名
    rootdir2 = r'.\\0断触数据集'  # 数据总文件夹名
    rootdir3 = r'.\\0延迟释放数据集'  # 数据总文件夹名
    rootdirA = r'..\nex高低温抵消实验切分后数据_长按部分_汇总'
    rootdir = r'.\20190423 切片前数据'
    rootdir = r'.\20190427_求差验证数据'
    rootdir = r'.\20190427_baseline不跟求差验证数据'
    rootdir = r'.\20190427_v101'
    rootdir = r'.\20190427_v102'

    # 运行其中一个函数
    # __analysis_supervised_data(rootdir2, compare_baseline=False)
    # __statistics(rootdir=rootdirA)
    __analysis_unsupervised_data(rootdir, compare_baseline=True, compare_forceflag=True, compare_stable_state=False)


if __name__ == '__main__':
    test_baseline()
