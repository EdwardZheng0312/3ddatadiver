import pandas as pd
import numpy as np
import loaddata
import zdirection

def testzdirection():
    z_direction, Z_dir = zdirection.zdirection("PHASEdata.csv","up")
    try:
        zdirection.zdirection(123,"up") or zdirection.zdirection("PHASEdata.csv",123)
    except(Exception):
        pass
    else:
        raise Exception("Input error. String-string expected")
        
    assert len(Z_dir)==500, 'z error'
   
    return