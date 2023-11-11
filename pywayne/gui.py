# !/usr/bin/env python
# -*- coding:utf-8 -*-
""" 
@author: Wang Ye (Wayne)
@file: gui.py
@time: 2022/03/01
@contact: wangye@oppo.com
@site: 
@software: PyCharm
# code is far away from bugs.
"""

import logging

try:
    import win32api
    import win32clipboard
    import win32con
    import win32gui
    import win32process
except ImportError:
    logging.warn('Try install pywin32 on your computer from below website:\n'
                 'http://www.lfd.uci.edu/~gohlke/pythonlibs')

try:
    import ctypes
    import ctypes.wintypes
except:
    logging.warn('You should use ctypes only in Windows')

try:
    from pykeyboard import PyKeyboard
    from pymouse import PyMouse
except ImportError:
    logging.warn('Try pip install pyuserinput and also have pyHook installed on your computer from below website:\n'
                 'http://www.lfd.uci.edu/~gohlke/pythonlibs')

try:
    import pyautogui
except ImportError:
    logging.warn('Try pip install pyautogui')

try:
    import pyttsx3
except ImportError:
    logging.warn('Try pip install pyttsx3')


class GlobalHotKeys:
    """
    Register a key using the register() method, or using the @register decorator
    Use listen() to start the message pump

    Author:   wangye
    Datetime: 2019/5/17 11:00

    Example:
    g = GlobalHotKeys()

    @GlobalHotKeys.register(GlobalHotKeys.VK_F1, GlobalHotKeys.MOD_SHIFT)
    def shift_f1():
        print('hello world')

    # Q and ctrl will stop message loop
    GlobalHotKeys.register(GlobalHotKeys.VK_Q, 0, False)
    GlobalHotKeys.register(GlobalHotKeys.VK_C, GlobalHotKeys.MOD_CTRL, False)

    # start main loop
    GlobalHotKeys.listen()
    """

    key_mapping = []
    user32 = None

    try:
        user32 = ctypes.windll.user32
        MOD_ALT = win32con.MOD_ALT
        MOD_CTRL = win32con.MOD_CONTROL
        MOD_CONTROL = win32con.MOD_CONTROL
        MOD_SHIFT = win32con.MOD_SHIFT
        MOD_WIN = win32con.MOD_WIN
    except:
        pass

    def __init__(self):
        self._include_alpha_numeric_vks()
        self._include_defined_vks()

    @classmethod
    def _include_defined_vks(cls):
        for item in win32con.__dict__:
            item = str(item)
            if item[:3] == 'VK_':
                setattr(cls, item, win32con.__dict__[item])

    @classmethod
    def _include_alpha_numeric_vks(cls):
        for key_code in (list(range(ord('A'), ord('Z') + 1)) + list(range(ord('0'), ord('9') + 1))):
            setattr(cls, 'VK_' + chr(key_code), key_code)

    @classmethod
    def register(cls, vk, modifier=0, func=None):
        """
        vk is a windows virtual key code
         - can use ord('X') for A-Z, and 0-1 (note uppercase letter only)
         - or win32con.VK_* constants
         - for full list of VKs see: http://msdn.microsoft.com/en-us/library/dd375731.aspx

        modifier is a win32con.MOD_* constant

        func is the function to run.  If False then break out of the message loop
        """

        # Called as a decorator?
        if func is None:
            def register_decorator(f):
                cls.register(vk, modifier, f)
                return f

            return register_decorator
        else:
            cls.key_mapping.append((vk, modifier, func))

    @classmethod
    def listen(cls):
        """
        Start the message pump
        """

        for index, (vk, modifiers, func) in enumerate(cls.key_mapping):
            # cmd 下没问题, 但是在服务中运行的时候抛出异常
            if not cls.user32.RegisterHotKey(None, index, modifiers, vk):
                raise Exception('Unable to register hot key: ' + str(vk) + ' error code is: ' + str(
                    ctypes.windll.kernel32.GetLastError()))

        try:
            msg = ctypes.wintypes.MSG()
            while cls.user32.GetMessageA(ctypes.byref(msg), None, 0, 0) != 0:
                if msg.message == win32con.WM_HOTKEY:
                    (vk, modifiers, func) = cls.key_mapping[msg.wParam]
                    if not func:
                        break
                    func()

                cls.user32.TranslateMessage(ctypes.byref(msg))
                cls.user32.DispatchMessageA(ctypes.byref(msg))

        finally:
            for index, (vk, modifiers, func) in enumerate(cls.key_mapping):
                cls.user32.UnregisterHotKey(None, index)


class GuiOperation:
    """
    Using package pywin32 to do some gui operations.

    Author:   wangye
    Datetime: 2019/5/18 22:00

    example:
    gui = GuiOperation()
    notepad = gui.find_window('tt')[0]
    gui.bring_to_top(notepad)
    time.sleep(2)
    st_test_software = gui.find_window('ST')[0]
    gui.bring_to_top(st_test_software)

    for h in gui.get_child_windows(st_test_software):
        ttl, cls = gui.get_windows_attr(h)
        print(ttl, cls)
        if '&Read' in ttl:
            print('-------')
            left, top, right, bottom = gui.get_window_rect(h)
            print(left, top, right, bottom)
            gui.mouse.move((left + right) // 2, (top + bottom) // 2)
            time.sleep(0.2)
            gui.change_window_name(h, 'shit')

    for proc in psutil.process_iter():
        try:
            pinfo = proc.as_dict(attrs=['pid', 'name'])
        except psutil.NoSuchProcess:
            pass
        else:
            print(pinfo)
    """

    def __init__(self):
        self.mouse = PyMouse()
        self.keyboard = PyKeyboard()
        # 系统常量，标识最高权限打开一个进程
        PROCESS_ALL_ACCESS = (0x000F0000 | 0x00100000 | 0xFFF)

    def find_window(self, *key):
        titles = set()

        def loop_windows(hwnd, _):
            if win32gui.IsWindow(hwnd) \
                    and win32gui.IsWindowEnabled(hwnd) \
                    and win32gui.IsWindowVisible(hwnd):
                titles.add(win32gui.GetWindowText(hwnd))

        win32gui.EnumWindows(loop_windows, 0)
        wanted_window_handles = [win32gui.FindWindow(None, t) for t in titles if all([k in t for k in key])]
        return wanted_window_handles

    def get_windows_attr(self, hwnd):
        if hwnd:
            return win32gui.GetWindowText(hwnd), \
                   win32gui.GetClassName(hwnd)
        return '', ''

    def maximize_window(self, hwnd):
        if hwnd:
            win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)

    def bring_to_top(self, hwnd):
        if hwnd:
            win32gui.ShowWindow(hwnd, win32con.SW_SHOW)
            win32gui.BringWindowToTop(hwnd)
            try:
                win32gui.SetForegroundWindow(hwnd)
            except:
                pass

    def close_window(self, hwnd):
        if hwnd:
            win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)

    def get_window_rect(self, hwnd):
        return win32gui.GetWindowRect(hwnd)

    def get_child_windows(self, hwnd):
        hwnd_child_list = set()
        stack = [hwnd]
        while stack:
            s = stack.pop()
            if s in hwnd_child_list:
                continue
            sub_hwnd_child_list = []
            try:
                win32gui.EnumChildWindows(
                    s, lambda h, p: p.append(h), sub_hwnd_child_list
                )
                [stack.append(sh) for sh in sub_hwnd_child_list]
                [hwnd_child_list.add(sh) for sh in sub_hwnd_child_list]
            except:
                continue
        return list(hwnd_child_list)

    def change_window_name(self, hwnd, new_name):
        win32api.SendMessage(hwnd, win32con.WM_SETTEXT, 0, new_name)
