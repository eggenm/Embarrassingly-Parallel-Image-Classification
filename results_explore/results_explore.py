# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 10:15:33 2019

@author: rheil
"""

import dirfuncs
import pandas as pd
import statsmodels.formula.api as smf
dropbox_dir = dirfuncs.guess_dropbox_dir()
data_dir = dropbox_dir + 'HCSproject\\data\\PoC\\'
result_csv = data_dir + 'result.12232019.csv'

result_df = pd.read_csv(result_csv)
mod = smf.ols(formula="two_class_score ~ bands + class_scheme", data=result_df)
res = mod.fit()
print(res.summary())