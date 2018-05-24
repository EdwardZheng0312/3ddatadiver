import pandas as pd
import numpy as np
import load_data

def test_load_data():
    data, z, z_approach, z_retract = load_data.load_data("PHASEdata.csv")
    try:
        load_data.load_data(123)
    except(Exception):
        pass
    else:
        raise Exception("Input error. String expected")
    assert data.shape == (1000,2305), 'data error'
    assert len(z) == 1000, 'z error'
    assert len(z_approach) == 500, 'z_appraoch error'
    assert len(z_retract) == 500, 'z_retract error'
    return