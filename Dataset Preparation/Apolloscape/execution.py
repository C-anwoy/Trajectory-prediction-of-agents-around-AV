# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 20:59:57 2021

@author: ANWOY
"""

import numpy as np
from preprocess_apolloscape import *


DATA_DIR="C:\\Users\\ANWOY\\Desktop\\APOLLOSCAPE\\asdt_sample_ trajectory\\"
DATA_DIR_TEST="C:\\Users\\ANWOY\\Desktop\\APOLLOSCAPE\\prediction_test\\"

save_dir_train='C:\\Users\\ANWOY\\Desktop\\processed_APOL\\train\\'
save_dir_val='C:\\Users\\ANWOY\\Desktop\\processed_APOL\\val\\'
save_dir_test='C:\\Users\\ANWOY\\Desktop\\processed_APOL\\test\\'

for i in range(1,8):
    format_apolloscape_main(DATA_DIR, DATA_DIR_TEST, 'train', save_dir_train, i)

for i in range(8,10):
    format_apolloscape_main(DATA_DIR, DATA_DIR_TEST, 'val', save_dir_val, i)

format_apolloscape_main(DATA_DIR, DATA_DIR_TEST, 'test', save_dir_test, 1)

