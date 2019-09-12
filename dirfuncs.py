# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 14:33:54 2019

@author: rheil
"""

import os

def check_possibilities(possibilities):
    """
    Parameters
    ----------
    possibilities : list
        Strings specifying possibly directories to check for existence.
        
    Returns
    -------
    p: string
        Path with a match        
    """

    for p in possibilities:
        if os.path.exists(p):
            return p

    return None
    
    
def guess_dropbox_dir():
    """
    Looks for the folder where dropbox data lives.
    """
    possibilities = ['D:\\cloud\\Dropbox\\collaborations\\',  # Robert's dropbox
                     'C:\\Users\\ME\\Dropbox\\'] # Michael's dropbox
    return check_possibilities(possibilities)

