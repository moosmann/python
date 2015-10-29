# -*- coding: utf-8 -*-
"""
Created on Wed May 20 13:34:55 2015

@author: jmoosmann
"""

def test():
    return 1, 2

def test2():
    return (1,2)

test()
x,y = test()
x2,y2 = test2()