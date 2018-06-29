import pandas as pd
import numpy as np
import load_data
import Z_direction

def test_Z_direction():
    z_direction, Z_dir = Z_direction.Z_direction("PHASEdata.csv","up")
    try:
        Z_direction.Z_direction(123,"up") or Z_direction.Z_direction("PHASEdata.csv",123)
    except(Exception):
        pass
    else:
        raise Exception("Input error. String-string expected")
        
    assert len(Z_dir)==500, 'z error'
   
    return