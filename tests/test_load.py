# Brokeback way to fix the problem, need to figure out why I am getting these errors...
import sys

sys.path.append('../')

import unittest
import implementation.load as load


class TestLoad(unittest.TestCase):
    @classmethod
    def setUpClass(cls):  # it will run before everything
        print('setUpClass')

    @classmethod
    def tearDownClass(cls):
        print('tearDownClass')

    def test_load_fmri(self):
        data = load.load_fmri(task="MOTOR")
        n_subjects, n_regions, n_timecourses = data.shape

        self.assertEqual(n_subjects, 100)
        self.assertEqual(n_regions, 379)

        with self.assertRaises(AssertionError):
            load.load_fmri('hello')

    def test_load_task_paradigms(self):
        data = load.load_task_paradigms(task='MOTOR')
        n_subjects = len(data)

        self.assertEqual(n_subjects, 100)

        with self.assertRaises(AssertionError):
            load.load_task_paradigms('hello')

    def test_load_hrf_function(self):
        hrf_function = load.load_hrf_function()

        self.assertEqual(len(hrf_function), 45)

    def test_separate_conditions(self):
        task_paradigms = load.load_task_paradigms(task='MOTOR')
        task_paradigms_one_hot = load.separate_conditions(task_paradigms)

        self.assertEqual(task_paradigms_one_hot.shape[1], 6)

    def test_do_one_hot(self):
        task_paradigms = load.load_task_paradigms(task='MOTOR')
        task_paradigms_one_hot = load.do_one_hot(task_paradigms[0].squeeze())

        self.assertEqual(task_paradigms_one_hot.shape[0], 6)
        self.assertEqual(task_paradigms_one_hot.shape[1], 284)
        self.assertEqual(task_paradigms_one_hot.max(), 1)

    def test_do_convolution(self):
        task_paradigms = load.load_task_paradigms(task='MOTOR')
        task_paradigms_one_hot = load.separate_conditions(task_paradigms)
        hrf = load.load_hrf_function()
        task_paradigms_conv = load.do_convolution(task_paradigms_one_hot, hrf)

        self.assertEqual(task_paradigms_conv.shape[0], 100)
        self.assertEqual(task_paradigms_conv.shape[1], 6)
        self.assertEqual(task_paradigms_conv.shape[2], 284)
        self.assertAlmostEqual(task_paradigms_conv[0, 0, 105], 0.58045, places=4)


if __name__ == '__main__':
    unittest.main()
