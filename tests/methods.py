# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 15:54:16 2015

@author: jmoosmann
"""

import unittest
import time
import re


# __all__ = [ "__", "___", "____", "_____", "Koan" ]


class Pizza(object):
    def __init__(self, size):
        self.size = size

    def get_size(self):
        return self.size

    @staticmethod
    def mix_ingredients(x, y):
        return x + y

    def cook(self):
        return self.mix_ingredients(self.cheese, self.vegetables)


class TestMethods(unittest.TestCase):
    """Test case to learn abstract, class, and static methods"""

    def assertMatch(self, pattern, string, msg=None):
        """
        Throw an exception if the regular expresson pattern is matched
        """
        # Not part of unittest, but convenient for some koans tests
        m = re.search(pattern, string)
        if not m or not m.group(0):
            raise self.failureException, \
                (msg or '{0!r} does not match {1!r}'.format(pattern, string))

    def assertNoMatch(self, pattern, string, msg=None):
        """
        Throw an exception if the regular expresson pattern is not matched
        """
        m = re.search(pattern, string)
        if m and m.group(0):
            raise self.failureException, \
                (msg or '{0!r} matches {1!r}'.format(pattern, string))

    def setUp(self):
        # Timing
        self.start_time = time.time()

    def tearDown(self):
        # Timing
        t = time.time() - self.start_time
        print "%s: %.3f" % (self.id(), t)

    def test_unbound_method(self):
        """A method is a function that is stored as a class attribute."""
        a = Pizza.get_size
        self.assertEqual(type(a).__name__, 'instancemethod')
        self.assertEqual(a.__str__(), '<unbound method '
                                      'Pizza.get_size>')

    def test_call_unbound_method(self):
        """Calling a method that is not bound to an instance raises an error."""
        try:
            Pizza.get_size()
        except Exception as ex:
            pass
        self.assertEqual(type(ex), TypeError)
        self.assertEqual(
            ex.args.__str__(),
            "('unbound method get_size() must be called with Pizza instance as first argument (got nothing instead)',)"
            )

    def test_call_unbound_method_with_instance_argument(self):
        """A method wants an instance as first argument. """
        a = Pizza.get_size(Pizza(10))
        self.assertEqual(a, 10)

    def test_call_bound_method(self):
        """Python binds methods from class CLASS to any instance of this
        class. Since the method is bound to the instance of the class,
        the self argument is automatically by the class instance and thus
        does not need to be provided. """
        a = Pizza(42).get_size
        print a
        print a.__self__
        self.assertEqual(a, a.__self__.get_size)
        # self.assertMatch('bound method Pizza.get_size', a)
        a = Pizza(10).get_size()
        self.assertEqual(a, 10)

    def test_static_methods(self):
        a = Pizza(10).cook
        self.assertFalse(Pizza(10).cook == Pizza(10).cook)


class Init(object):
    d = 'd'

    def __init__(self):
        self.c = 0
        self.a = self._a()
        self.b = 1


    def _a(self):
        print "Call of method 'self._a()' via __init__."
        self.c += 1
        return 'a'

    def f(self):
        return self.d

    def g1(self):

        self._g()

    def g2(self):

        return self._g()

    def _g(self):

        return 'yes'


class TestInit(unittest.TestCase):

    def test_initialization(self):
        t = Init()
        self.assertEqual(t.f(), 'd')

        print t.a, t.b, t.c, t.a, t.c

    def test_return(self):

        t = Init()
        g1 = t.g1()
        g2 = t.g2()
        self.assertEqual(g1, None)
        self.assertEqual(g2, 'yes')

        print 'Finished test_return.'

        


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestInit)
    unittest.TextTestRunner(verbosity=0).run(suite)
    # unittest.main()
