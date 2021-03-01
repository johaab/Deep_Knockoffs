import unittest

from implementation import glm
from implementation.load import load_fmri, load_task_paradigms, load_hrf_function


class TestLoad(unittest.TestCase):
    @classmethod
    def setUpClass(cls):  # it will run before everything
        print('setUpClass')

    @classmethod
    def tearDownClass(cls):
        print('tearDownClass')

    def test_glm(self):
        # test that the correlation with the true betas is >0.99
        fMRI = load_fmri('MOTOR')
        task_paradigms = load_task_paradigms('MOTOR')
        hrf = load_hrf_function()
        activations, controlled_act, betas, tvalues, uncontrolled_betas, controlled_betas = glm.glm(fMRI, task_paradigms, hrf)

        self.assertEqual(activations.shape[2], 5)
        self.assertEqual(betas.shape[2], 5)
        self.assertEqual(activations[0, 2, 0], 1)
        self.assertAlmostEqual(betas[0, 2, 0], -0.62251, places=4)


if __name__ == '__main__':
    unittest.main()
