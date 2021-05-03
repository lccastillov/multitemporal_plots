## ---------------------------
##
## Script name: Multi_temporal_monitoring.py
##
## Purpose of script:
#                   1. Retrieve typical temporal patterns of vegetation indices (VIs) or satellite products using historical \
#                   data from multiple plots at different ages.
#                   2. Compare current VI values per plot with typical values based on the analysis of historical data

##
## Author: Liliana Castillo Villamor. Affiliation: Aberystwyth University

##
## Date Created: 03/05/2021
##
## Copyright (c) Liliana Castillo Villamor, 2021
## Email: lic42@aber.ac.uk
##
## ---------------------------
##
## Notes:
## the functions used in this script are kept in the file functions_multitemporal_monitoring.py
## Description of fields in the input files is presented in "input_files_description"
##
## ---------------------------
##
##
## --------------------------
#################################################################
###### Import Functions #####

import time
import datetime
import random
import csv
import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
from numpy import genfromtxt
import os
import warnings



warnings.simplefilter(action='ignore', category=FutureWarning)


