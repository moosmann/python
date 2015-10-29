# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 14:17:58 2015

@author: jmoosmann
"""

# Imports for common Python 2/3 codebase
from __future__ import (unicode_literals, print_function, division,
                        absolute_import)
from builtins import super
# from future import standard_library

__metaclass__ = type

# Basic working principle of super(). 
class A(object):
    def foo(self):
        print('A')
 
class B(A):
    def foo(self):
        print('B')
        super(B, self).foo()
 
class C(A):
    def foo(self):
        print('C')
        super(C, self).foo()
 
class D(B,C):
    def foo(self):
        print('D')
        super(D, self).foo()
        print(D.__mro__)
        
d = D()
d.foo()

# Example: create a subclass for extending a method from one of the builtin 
# classes. Before the introduciton of super(), the call would have needed to be
# hardwired with dict.__setitem__(self, key, value) instead of using super() as
# computed indirect reference
class LoggingDict(dict):
    def __setitem__(self, key, value):
        logging.info('Settingto %r' % (key, value))
        super().__setitem__(key, value)




class Shape:
    def __init__(self, shapename, **kwds):
        self.shapename = shapename
        super().__init__(**kwds)        

class ColoredShape(Shape):
    def __init__(self, color, **kwds):
        self.color = color
        super().__init__(**kwds)

cs = ColoredShape(color='red', shapename='circle')


class Root:
    def draw(self):
        # the delegation chain stops here
        assert not hasattr(super(), 'draw')

class Shape(Root):
    def __init__(self, shapename, **kwds):
        self.shapename = shapename
        super().__init__(**kwds)
    def draw(self):
        print('Drawing.  Setting shape to:', self.shapename)
        super().draw()

class ColoredShape(Shape):
    def __init__(self, color, **kwds):
        self.color = color
        super().__init__(**kwds)
    def draw(self):
        print('Drawing.  Setting color to:', self.color)
        super().draw()

cs = ColoredShape(color='blue', shapename='square')
cs.draw()