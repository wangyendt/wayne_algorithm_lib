AHRS 工具函数 (tools)
======================

.. automodule:: pywayne.ahrs.tools
   :members:
   :undoc-members:

本模块提供了用于姿态估计的工具函数，主要基于四元数数学原理来解算物体的整体旋转角、航向角及倾斜角。模块中主要包含两个函数，分别用于四元数分解和姿态补偿。

.. autofunction:: pywayne.ahrs.tools.quaternion_decompose
   :noindex:

**示例**::

   >>> from pywayne.ahrs.tools import quaternion_decompose
   >>> import numpy as np
   >>> q = np.array([np.sqrt(2)/2, np.sqrt(2)/2, 0, 0])
   >>> print(quaternion_decompose(q))

.. autofunction:: pywayne.ahrs.tools.quaternion_roll_pitch_compensate
   :noindex:

**示例**::

   >>> from pywayne.ahrs.tools import quaternion_roll_pitch_compensate
   >>> import numpy as np
   >>> q = np.array([0.989893, -0.099295, 0.024504, -0.098242])
   >>> print(quaternion_roll_pitch_compensate(q)) 