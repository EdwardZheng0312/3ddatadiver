import unittest


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()


def test_bin_array():
    test1 = np.random.random_integers(0, 9 + 1, size=(500, 20, 20))
    test2 = np.random.random_integers(0, 9 + 1, size=(500, 20, 20))
    test3 = np.random.random_integers(0, 9 + 1, size=(20, 20))
    result1, result2, result3 = bin_array(test1, test3, test2)

    try:
        correct_slope(test1)
    except Exception:
        pass
    else:
        raise Exception("Did not catch array ordering error.")

    assert 70 < len(result1) < len(test1), "Linearized not a reasonable length\
                                                    check that arraytotcorr values are between\
                                                    -0.1 and 5"
    assert np.isnan(np.sum(result2)) == True, "Max extension not normalized properly. Check\
                                            that correct target array is selected."
    assert np.isnan(np.sum(result3)) == True, "Max extension not normalized properly. Check\
                                            that correct target array is selected."