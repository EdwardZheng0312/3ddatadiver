import numpy as np
import pandas as pd
import load_data
import create_pslist

def test_create_pslist():
    pslist = create_pslist.create_pslist("PHASEdata.csv", 48, 48)
    
    try:
        create_pslist.create_pslist(12, 48, 48)
    except(Exception):
        pass
    else:
        raise Exception("Input type error. String-integer-integer expected")
    
    try:
        create_pslist.create_pslist("PHASEdata.csv", "a", 48)
    except(Exception):
        pass
    else:
        raise Exception("Input type error. String-integer-integer expected")
        
    try:
        create_pslist.create_pslist("PHASEdata.csv", 48, "a")
    except(Exception):
        pass
    else:
        raise Exception("Input type error. String-integer-integer expected")
        
    assert len(pslist) == 1000, 'pslist length error'

    return