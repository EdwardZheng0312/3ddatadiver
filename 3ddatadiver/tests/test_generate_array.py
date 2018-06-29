def test_generate_array():
    test1 = np.random.random_integers(0, 9 + 1, size=(500, 20, 20))
    try:
        correct_slope(test1)
    except Exception:
        pass
    else:
        raise Exception("Did not catch array ordering error.")
    test2 = np.random.random_integers(0, 9 + 1, size=(500, 20, 20, 20))
    try:
        correct_slope(test2)
    except Exception:
        pass
    else:
        raise Exception("Did not catch check transpose error.")