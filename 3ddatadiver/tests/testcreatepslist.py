import numpy as np
import pandas as pd
import loaddata
import createpslist

def testcreatepslist():
    pslist = createpslist.createpslist("PHASEdata.csv", 48, 48)
    
    try:
        createpslist.createpslist(12, 48, 48)
    except(Exception):
        pass
    else:
        raise Exception("Input type error. String-integer-integer expected")
    
    try:
        createpslist.createpslist("PHASEdata.csv", "a", 48)
    except(Exception):
        pass
    else:
        raise Exception("Input type error. String-integer-integer expected")
        
    try:
        createpslist.createpslist("PHASEdata.csv", 48, "a")
    except(Exception):
        pass
    else:
        raise Exception("Input type error. String-integer-integer expected")
        
    assert len(pslist) == 1000, 'pslist length error'

    return