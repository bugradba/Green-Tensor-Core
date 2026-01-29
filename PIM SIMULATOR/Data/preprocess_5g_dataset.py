import pandas as pd
import numpy as np
from datetime import datetime
import pyshark

class FiverGDatasetPreproessor:
     """
    Western-OC2-Lab 5G dataset'ini ECO-PIM için hazırlar.
    
    Input: Dataset2.csv veya Dataset1.pcapng
    Output: ML-ready anomaly detection dataset
    """
     
