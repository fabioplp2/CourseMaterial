import pygeostat as gs
import unittest

'''
unittest, pytest and doctest are the three major tools for testing in python.
For unittest, the test runner is called
    >>> python -m unittest -v
    >>> python .\test_griddef.py -v
    In order to use doctest, additional code is required
For pytest, the test runner is called
    >>> python -m pytest -v
    To get doc string test using doc test
    >>> python -m pytest --doctest-modules

Note: -v is used to get a list of tests being executed.
'''

class GridDefTest(unittest.TestCase):


    '''
        This class is used to implement unittests for pygeostat griddef class
    '''

    def setUp(self):
        '''
        Method called to prepare the test fixture. This is called immediately before calling the test method; 
        other than AssertionError or SkipTest, any exception raised by this method will be considered 
        an error rather than a test failure. The default implementation does nothing.
        '''
        griddef_str = '''120 5.0 10.0 
                        110 1205.0 10.0 
                        1 0.5 1.0'''
        self.griddef = gs.GridDef(griddef_str)

    def test_nx(self):
        self.assertEqual(self.griddef.nx, 120)

    def test_ny(self):
        self.assertEqual(self.griddef.ny, 110)

    def test_orientation(self):
        with self.assertRaises(ValueError):
            _ = self.griddef.outline_pts(orient='zy')

    def tearDown(self):
        '''
        Method called immediately after the test method has been called and the result recorded. 
        This is called even if the test method raised an exception, so the implementation in subclasses may 
        need to be particularly careful about checking internal state. 
        Any exception, other than AssertionError or SkipTest, raised by this method will be 
        considered an additional error rather than a test failure (thus increasing the total number of reported errors). 
        This method will only be called if the setUp() succeeds, regardless of the outcome of the test method. 
        The default implementation does nothing.
        '''
        pass


if __name__ == '__main__':
    unittest.main()
