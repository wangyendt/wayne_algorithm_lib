# This is a useful tool mainly includes:

### 1. Decorator
a. (decorator) func_timer: calculate a function's running time

b. (decorator) func_timer_batch: calculate a function's accumulated running time

c. (decorator) maximize_figure: maximize specific figure

d. (decorator) singleton: singleton for object in python

e. (decorator) binding_press_release: binding some keyboards and mouse operations to a specific figure

### 2. Data Structure
a. (class) ConditionTree: designed for debugging state-machine-like module including amounts of if-else

b. (class) UnionFind: typical union find set

### 3. Useful Tools
a. (function) list_all_files: list all the files in all sub-directories, can specify keywords including and outliers excluding

b. (function) count_file_lines: count the line number of file of arbitary size

c. (function) leader_speech: generate some random leader_speech

d. (class) GlobalHotKeys: used for keyboard listening

e. (class) GuiOperation: used for gui operations

f. (class) XmlIO: used to read/write xml files

### 4. Data Processing
a. (function) peak_det: detect peak with O(n) time complexity, easy to extend

b. (function) butter_bandpass_filter: typical butterworth filter

c. (class) FindLocalExtremum: find local extremum

d. (class) CurveSimilarity: compare two curves' similarity

e. (function) find_extremum_in_sliding_window: find extremum in sliding window in O(n) time complexity

### 5. Mathematics

a. (function) get_all_factors: get all the factors of a given natural number

b. (function) digit_counts: count specific number's appearance from 1 to n

c. (function) karatsuba_multiplication: long number multiplication


For more details, take a look at the details in tools.py

# Installation
Try to run "pip install pywayne" in your commond line tool.

# How to use
from pywayne.tools import *

files = list_all_files(path_to_root)

...

# P.S.
Feel free to contact me whenever you need help using this tool.

e-mail: wang121ye@hotmail.com

leetcode: http://leetcode.com/wangyehope

github: http://github.com/wangyendt

Pull-requests are always welcomed.

Hopefully this may help you somewhere!
