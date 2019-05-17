import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

from ndtpy.tools import list_all_files
from ndtpy.tools import LoadData
from ndtpy.fw.utils import FindLocalExtremeValue

from ndtpy.fw.baseline_diff_v0 import BaseLine, leave_type_dict

# from baseline_diff_v1 import BaseLine, leave_type_dict

plt.rcParams['font.family'] = 'FangSong'
plt.rcParams['axes.unicode_minus'] = False
ORDER1 = 16
CH_NUMS = 6


def run_example(rawdata, start_frame=100, plot_out_flag=False, plot_slow_release_flag=False, suptitle='',
                forceflag_true=None, statistic_flag=False, baseline_fw=None, stable_state_fw=None, forceflag_fw=None):
    """
    运行例子，
    可以直接运行run_example函数
    也可以参考此函数，重写对force_flag_detect_I_II和baseline_tracking的调用，将更加灵活
    :param start_frame: 开始计算的点,必须大于50
    :param plot_out_flag:   是否 打印基线和forceflag计算结果
    :param plot_slow_release_flag: plot缓慢释放条件触发点，若为真，则一定打印基线和forceflag计算结果
    :param forceflag_true: forcelfag的参考真值，若不为None, 则plot
    :param statistic_flag: 切片数据统计时用，忽略所有的画图。为真时forceflag_true必须存在
    :param suptitle 图的名字,画图时才有意义
    :param baseline_fw       分析baseline追踪趋势和stable_state时用
    :param stable_state_fw   分析baseline追踪趋势和stable_state时用
    :return:
             只有在statistic_flag为True时有意义
             statistic_type: 0. 正常释放
                             1. 延迟释放
                             2. 无法释放
                             3. 断触
    """
    statistic_type = 2  # 默认为无法释放
    bsl = BaseLine(rawdata)
    rd_diff = np.zeros_like(rawdata)
    rd_diff[:, 0] = rawdata[:, 0] - rawdata[:, 2]
    rd_diff[:, 1] = rawdata[:, 1] - rawdata[:, 3]
    rd_diff[:, 2] = rawdata[:, 2] - rawdata[:, 0]
    rd_diff[:, 3] = rawdata[:, 3] - rawdata[:, 1]
    rd_diff[:, 4] = rawdata[:, 4]
    rd_diff[:, 5] = rawdata[:, 5]
    bsl_diff = BaseLine(rd_diff)
    assert (start_frame > 3 * ORDER1 + 2)
    if (not statistic_flag) and (plot_slow_release_flag or plot_out_flag):
        plt.figure()
        plt.suptitle(suptitle)
    # 主逻辑
    for i in range(start_frame, np.shape(bsl.rawdata)[0]):
        # 1. 更新 forceflag
        if bsl.forceflag_sw[i - 1] == 0:
            bsl.forceflag_sw[i], bsl.leave_type[i] = \
                bsl.force_flag_detect_I_II(bsl.rawdata[(i - 3 * ORDER1 - 2):i + 1],
                                           bsl.baseline_sw[i - 1],
                                           bsl.fbase_sw[i - 1],
                                           bsl.forceflag_sw[i - 1],
                                           temperature_err_flag=1, sig_err_flag=0)
            bsl_diff.deepcopy_state(bsl)
            bsl_diff.forceflag_sw[i] = bsl.forceflag_sw[i]
            bsl_diff.leave_type[i] = bsl.leave_type[i]
        else:  # bsl.forceflag[i-1] == 1
            bsl_diff.forceflag_sw[i], bsl_diff.leave_type[i] = \
                bsl_diff.force_flag_detect_I_II(bsl_diff.rawdata[(i - 3 * ORDER1 - 2):i + 1],
                                                bsl_diff.baseline_sw[i - 1],
                                                bsl_diff.fbase_sw[i - 1],
                                                bsl_diff.forceflag_sw[i - 1],
                                                temperature_err_flag=1, sig_err_flag=0)

            bsl.forceflag_sw[i] = bsl_diff.forceflag_sw[i]
            bsl.leave_type[i] = bsl_diff.leave_type[i]
            # 统计离手原因,并return
            if bsl.forceflag_sw[i] == 0 and statistic_flag:
                assert (forceflag_true is not None)
                assert (forceflag_true.shape == bsl.forceflag_sw.shape)
                true_end_id = np.where(np.diff(forceflag_true) == -1)[0]
                assert (true_end_id.shape[0] == 1)
                if i > true_end_id + 40:  # 延迟释放
                    statistic_type = 1
                elif i < true_end_id - 30:  # 断触
                    statistic_type = 3
                else:
                    statistic_type = 0  # 正常离手
                return statistic_type
            # 打印leave_type
            if (not statistic_flag) and bsl.leave_type[i] > 0:
                print("index: %d, leave type:%4.1f,%s" % (i, bsl.leave_type[i], leave_type_dict[bsl.leave_type[i]]))
            # 离手1帧拉回基线
            if bsl.forceflag_sw[i] == 0:
                last_maxchs = bsl.state_dict['last_maxch']
                # print(last_maxchs)
                bsl.deepcopy_state(bsl_diff)
                last_maxch = last_maxchs[0]
                valley_pos = 5 - np.argmin(rawdata[i - 5:i, last_maxch])
                # 1帧拉系数
                for ch in range(0, 6):
                    if ch in [4, 5]:
                        bsl.baseline_sw[i, ch] = rawdata[i - 2, ch]
                        bsl.fbase_sw[i, ch] = rawdata[i - 2, ch] * 1024
                        bsl_diff.baseline_sw[i, ch] = bsl_diff.rawdata[i - 2, ch]
                        bsl_diff.fbase_sw[i, ch] = bsl_diff.rawdata[i - 2, ch] * 1024
                    else:
                        bsl.baseline_sw[i, ch] = bsl.rawdata[i - valley_pos, ch]
                        bsl.fbase_sw[i, ch] = bsl.rawdata[i - valley_pos, ch] * 1024
                        bsl_diff.baseline_sw[i, ch] = bsl_diff.rawdata[i - valley_pos, ch]
                        bsl_diff.fbase_sw[i, ch] = bsl_diff.rawdata[i - valley_pos, ch] * 1024
                continue
            bsl.deepcopy_state(bsl_diff)
        # 2. 更新 baseline
        bsl.baseline_sw[i], bsl.fbase_sw[i], bsl.stable_state_sw[i] = \
            bsl.baseline_tracking(bsl.rawdata[i - 2 * ORDER1 + 1:i + 1],
                                  bsl.forceflag_sw[i], bsl.fbase_sw[i - 1],
                                  temperature_err_flag=1)
        bsl_diff.baseline_sw[i], bsl_diff.fbase_sw[i], bsl_diff.stable_state_sw[i] = \
            bsl_diff.baseline_tracking(bsl_diff.rawdata[i - 2 * ORDER1 + 1:i + 1],
                                       bsl_diff.forceflag_sw[i], bsl_diff.fbase_sw[i - 1],
                                       temperature_err_flag=1)
    # 画图
    if (not statistic_flag) and (plot_slow_release_flag or plot_out_flag):
        assert (CH_NUMS == 6)
        for ch in range(CH_NUMS):
            plt.subplot(3, 2, ch + 1)
            plt.title('ch' + str(ch + 1))
            plt.plot(bsl.rawdata[:, ch] - 1 * bsl.rawdata[start_frame, ch], '--')
            plt.plot(bsl.baseline_sw[:, ch] - 1 * bsl.rawdata[start_frame, ch])
            plt.plot(bsl.forceflag_sw * np.max(bsl.rawdata[:, ch] - bsl.baseline_sw[:, ch]) * 0.5)
            plt.legend(('rawdata', 'baseline', 'forceflag'))
            if forceflag_true is not None:
                assert (forceflag_true.shape == bsl.forceflag_sw.shape)
                plt.plot(forceflag_true * np.max(bsl.rawdata[:, ch] - bsl.baseline_sw[:, ch]) * 0.6)
                plt.legend(('rawdata', 'baseline', 'forceflag', 'forceflag_true'))

        plt.figure()
        plt.suptitle('差分')
        for ch in range(CH_NUMS):
            plt.subplot(3, 2, ch + 1)
            if ch == 0:
                plt.title('ch1-ch3')
            elif ch == 1:
                plt.title('ch2-ch4')
            elif ch == 2:
                plt.title('ch3-ch1')
            elif ch == 3:
                plt.title('ch4-ch2')
            elif ch == 4:
                plt.title('ch5')
            elif ch == 5:
                plt.title('ch6')
            plt.plot(bsl_diff.rawdata[:, ch] - 1 * bsl_diff.rawdata[start_frame, ch], '--')
            plt.plot(bsl_diff.baseline_sw[:, ch] - 1 * bsl_diff.rawdata[start_frame, ch])
            plt.plot(bsl_diff.forceflag_sw * np.max(bsl_diff.rawdata[:, ch] - bsl_diff.baseline_sw[:, ch]) * 0.5)
            plt.legend(('rawdata', 'baseline', 'forceflag'))
            if forceflag_true is not None:
                assert (forceflag_true.shape == bsl_diff.forceflag_sw.shape)
                plt.plot(forceflag_true * np.max(bsl_diff.rawdata[:, ch] - bsl_diff.baseline_sw[:, ch]) * 0.6)
                plt.legend(('rawdata', 'baseline', 'forceflag', 'forceflag_true'))
        if (baseline_fw is None) and (stable_state_fw is None):
            plt.show()  # 两个图画完之后才show
    # 画图比较fw和sw的baseline和stable_state
    if (baseline_fw is not None) or (stable_state_fw is not None):
        plt.figure()
        plt.suptitle(suptitle + ' + compare')
        for ch in range(CH_NUMS):
            plt.subplot(3, 2, ch + 1)
            plt.title('ch' + str(ch + 1))
            plt.plot(bsl.rawdata[:, ch] - bsl.rawdata[start_frame, ch], '--')
            legend = ('rawdata',)
            if baseline_fw is not None:
                assert (baseline_fw.shape == bsl.baseline_sw.shape)
                plt.plot(bsl.baseline_sw[:, ch] - bsl.rawdata[start_frame, ch])
                plt.plot(baseline_fw[:, ch] - baseline_fw[start_frame, ch])
                legend += ('baseline', 'baseline_fw')
            if stable_state_fw is not None:
                assert (stable_state_fw.shape == bsl.stable_state_sw.shape)
                plt.plot(bsl.stable_state_sw[:, ch] * np.max(bsl.rawdata[:, ch] - bsl.baseline_sw[:, ch]) * 0.5)
                plt.plot((stable_state_fw[:, ch] > 0) * np.max(bsl.rawdata[:, ch] - bsl.baseline_sw[:, ch]) * 0.6)
                legend += ('stable_state', 'stable_state_fw')
            if forceflag_fw is not None:
                assert (forceflag_fw.shape == bsl.forceflag_sw.shape)
                plt.plot(bsl.forceflag_sw * np.max(bsl.rawdata[:, ch] - bsl.baseline_sw[:, ch]) * 0.5)
                plt.plot((forceflag_fw > 0) * np.max(bsl.rawdata[:, ch] - bsl.baseline_sw[:, ch]) * 0.6)
                legend += ('forceflag', 'forceflag_fw')
            plt.legend(legend)

        plt.show()
    return statistic_type


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
        statistic_type = run_example(rawdata=rawdata, start_frame=start, forceflag_true=forceflag_true,
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
        if compare_baseline:
            run_example(rawdata=rawdata, start_frame=start, plot_out_flag=True, suptitle=suptitle,
                        forceflag_true=forceflag_true, baseline_fw=baseline_fw)
        else:
            run_example(rawdata=rawdata, start_frame=start, plot_out_flag=True, suptitle=suptitle,
                        forceflag_true=forceflag_true)
    return None


def __analysis_unsupervised_data(rootdir, compare_baseline=False, compare_forceflag=False, compare_stable_state=False):
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
    for fid, path in enumerate(files[4:]):
        print('haha', fid, path)
        data, rawdata, baseline_fw, forceflag_fw, forcesig_fw, temperature_err_flag_fw, sig_err_flag, stable_state_fw \
            = data_loader.load_data(path)  # 导入完整文件
        start = 100
        suptitle = path.split('\\')[-1][:-4]
        bsl_fw = baseline_fw if compare_baseline else None
        ss_fw = stable_state_fw if compare_stable_state else None
        ff_fw = forceflag_fw if compare_forceflag else None
        run_example(rawdata, start_frame=start, plot_out_flag=True, suptitle=suptitle, baseline_fw=bsl_fw,
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

    # 运行其中一个函数
    # __analysis_supervised_data(rootdirA, compare_baseline=False)
    # __statistics(rootdir=rootdirA)
    __analysis_unsupervised_data(rootdir, compare_baseline=True, compare_forceflag=True, compare_stable_state=False)


if __name__ == '__main__':
    test_baseline()
