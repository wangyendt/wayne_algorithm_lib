import os
import shutil

from pywayne.tools import list_all_files
from pywayne.fw.utils import FindLocalExtremeValue, LoadData
from pywayne.fw.baseline.baseline_local_v0 import BaseLine


def __statistics(rootdir=r'..\\dataset_all', fid_bg=0, fid_ed=9999):
    def preparedir(dir0):
        if os.path.exists(dir0):
            shutil.rmtree(dir0)
        os.mkdir(dir0)
    # 准备文件夹
    preparedir('..\\0延迟释放数据集')
    preparedir('..\\0断触数据集')
    preparedir('..\\0不释放数据集')

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
        print(fid, total_case, '\n', path)
        # load 文件
        data, rawdata, baseline_fw, forceflag_fw, forcesig_fw, forceflag_true, temperature_err_flag_fw, sig_err_flag \
            = data_loader.load_data(path)  # 导入切片数据
        # 起始点设置
        _, local_gap, _, _, _ = flev.run(rawdata[:101, 5])
        start = int(101 - local_gap)
        # 初始化
        bsl = BaseLine(rawdata)
        statistic_type = bsl.run_example(start_frame=start, forceflag_true=forceflag_true, statistic_flag=True)
        # 统计
        if statistic_type==1:  # 延迟释放
            slow_release_case += 1
            shutil.copy(path, '..\\0延迟释放数据集\\' + path.split('\\')[-1])
        elif statistic_type==2:  # 断触
            early_stop_case += 1
            shutil.copy(path, '..\\0断触数据集\\' + path.split('\\')[-1])
        elif statistic_type == 3:
            not_release_case += 1
            shutil.copy(path, '..\\0不释放数据集\\' + path.split('\\')[-1])
    # 打印统计结果
    print('一共有%d cases,%d个延迟释放(%0.2f)%%，%d个不释放(%0.2f)%%,%d个断触(%0.2f)%%' %
          (total_case,
           slow_release_case, slow_release_case / (total_case+1e-8) * 100,
           not_release_case, not_release_case / (total_case+1e-8) * 100,
           early_stop_case, early_stop_case / (total_case+1e-8) * 100))


def __analysis_supervised_data(rootdir,compare_baseline=False):
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
        _, local_gap, _, _, _ = flev.run(rawdata[:101, 5])
        start = int(101 - local_gap)
        print('start_frame',start)
        bsl = BaseLine(rawdata)
        suptitle = path.split('\\')[-1]
        if compare_baseline:
            bsl.run_example(start_frame=start, plot_slow_release_flag=True, suptitle=suptitle,
                            baseline_fw=baseline_fw,forceflag_true=forceflag_true)
        else:
            bsl.run_example(start_frame=start, plot_slow_release_flag=True, suptitle=suptitle,
                            forceflag_true=forceflag_true)
    return None


def __analysis_unsupervised_data(rootdir,compare_baseline=False, compare_stable_state=False):
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
        print(fid, path)
        data, rawdata, baseline_fw, forceflag_fw, forcesig_fw, temperature_err_flag_fw, sig_err_flag, stable_state_fw \
            = data_loader.load_data(path)  # 导入完整文件
        start = 100
        bsl = BaseLine(rawdata)
        suptitle = path.split('\\')[-1]
        if compare_baseline and compare_stable_state:
            bsl.run_example(start_frame=start, plot_out_flag=True, suptitle=suptitle,
                            baseline_fw=baseline_fw, stable_state_fw=stable_state_fw)
        elif compare_baseline:
            bsl.run_example(start_frame=start, plot_out_flag=True, suptitle=suptitle,
                            baseline_fw=baseline_fw)
        elif compare_stable_state:
            bsl.run_example(start_frame=start, plot_out_flag=True, suptitle=suptitle,
                            stable_state_fw=stable_state_fw)
        else:
            bsl.run_example(start_frame=start, plot_out_flag=True, suptitle=suptitle)
    return


def test_baseline():
    # 选定文件夹
    rootdir = r'..\20190417_50℃实验数据_2'
    rootdir = r'..\\dataset_all'  # 数据总文件夹名
    # rootdir = r'..\slow_release\real_slow'  # 数据总文件夹名
    # rootdir = r'..\slow_release'  # 数据总文件夹名
    # rootdir = r'..\\0不释放数据集'  # 数据总文件夹名
    # rootdir = r'..\\0断触数据集'  # 数据总文件夹名
    # rootdir = r'..\\0延迟释放数据集'  # 数据总文件夹名
    # rootdir = '.'

    # 运行其中一个函数
    __analysis_supervised_data(rootdir, compare_baseline=False)
    # __analysis_unsupervised_data(rootdir, compare_baseline=True, compare_stable_state=True)
    #__statistics(rootdir)


if __name__ == '__main__':
    test_baseline()

