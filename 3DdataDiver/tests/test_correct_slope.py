import unittest


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()


def test_correct_slope():
    test1 = np.random.random_integers(0, 9 + 1, size=(500, 20, 20))
    try:
        correct_slope(test1)
    except Exception:
        pass
    else:
        raise Exception("Did not catch array ordering error.")

    test2 = np.reshape(test1, (500, 20, 20), order='F')
    try:
        correct_slope(test2)
    except Exception:
        pass
    else:
        raise Exception("Did not catch max extension error.")

    assert np.mean(threeD_array) < np.mean(arraytotcorr), \
        "Input array non-negative, check input array"