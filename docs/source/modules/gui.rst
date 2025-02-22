图形用户界面 (gui)
====================

本模块主要提供与图形用户界面相关的功能，旨在支持全局热键注册以及窗口操作管理。借助该模块，用户可以轻松实现快捷键监听、窗口查找与操作，从而实现自动化桌面操作和 GUI 应用控制。

GlobalHotKeys 类
------------------

.. py:class:: GlobalHotKeys

   GlobalHotKeys 类用于注册全局热键，可通过 @register 装饰器或者直接调用 register 方法来绑定按键事件。注册的热键在调用 listen() 方法后会被监听，从而触发相应的处理函数。
   
   **主要功能**:
   
   - 注册热键（支持组合键，如 Shift+F1、Ctrl+C 等）。
   - 在全局范围内监听键盘事件。
   - 可通过列表管理已注册的按键与回调函数。
   
   **示例**::
   
      >>> from pywayne.gui import GlobalHotKeys
      >>> g = GlobalHotKeys()
      >>> @GlobalHotKeys.register(GlobalHotKeys.VK_F1, GlobalHotKeys.MOD_SHIFT)
      ... def shift_f1():
      ...     print('Hello, World!')
      >>> 
      >>> # 启动热键监听（按 Q 或 Ctrl+C 可退出）
      >>> GlobalHotKeys.listen()

GuiOperation 类
-----------------

.. py:class:: GuiOperation

   GuiOperation 类提供了多种窗口操作功能，包括查找窗口、获取窗口属性、最大化窗口、置顶窗口、关闭窗口以及修改窗口标题等。该类适用于需要进行 GUI 自动化测试、窗口管理或系统级界面控制的应用场景。
   
   **主要方法**:
   
   - **find_window(self, *key)**: 查找符合指定关键字的窗口，并返回窗口句柄。
   - **get_windows_attr(self, hwnd)**: 获取指定窗口的属性信息。
   - **maximize_window(self, hwnd)**: 将指定窗口最大化。
   - **bring_to_top(self, hwnd)**: 使指定窗口置于顶层。
   - **close_window(self, hwnd)**: 关闭指定窗口。
   - **change_window_name(self, hwnd, new_name)**: 修改指定窗口的标题。
   
   **示例**::
   
      >>> from pywayne.gui import GuiOperation
      >>> gui_op = GuiOperation()
      >>> # 查找包含"记事本"的窗口
      >>> hwnd = gui_op.find_window('记事本')
      >>> if hwnd:
      ...     gui_op.maximize_window(hwnd)
      ...     gui_op.bring_to_top(hwnd)

--------------------------------------------------

通过上述示例，用户可以快速掌握 gui 模块中各工具的使用方法，有效地实现全局热键注册以及窗口操作控制，从而丰富桌面应用的交互体验。 