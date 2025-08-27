# # 基础工具模块 - 核心功能，应该总是可用
# from . import tools
# from . import dsp
# from . import maths
# from . import statistics
# from . import data_structure
# from . import plot
# from . import helper
# from . import crypto

# # 专业功能模块
# from . import ahrs
# from . import calibration
# from . import adb
# from . import cv
# from . import llm
# from . import vio
# from . import visualization

# # 通信与集成模块
# from . import lark_custom_bot
# from . import lark_bot
# from . import lark_bot_listener
# from . import tts
# from . import aliyun_oss
# from . import cross_comm

# 注释掉的模块（待实现或已废弃）
# from . import tt_api
# from . import ocs_api

__all__ = [
    'tools',
    'dsp',
    'gui',
    'maths',
    'statistics',
    'data_structure',
    'plot',
    'ahrs',
    'calibration',
    'lark_custom_bot',
    'lark_bot',
    'lark_bot_listener',
    'tts',
    'aliyun_oss',
    'helper',
    'cross_comm',
    'crypto',
    'adb',
    'cv',
    'llm',
    'vio',
    'visualization',
    # 'tt_api',
    # 'ocs_api',
]

name = 'pywayne'
