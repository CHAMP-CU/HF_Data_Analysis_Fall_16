# This file contains the functions that are needed for the champ.py script to run
# for the Fall 2016 code.
# Authors: Thomas Jeffries
# Created: 20161110
# Modified: 20161110

import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob
from pandas import DataFrame
import pandas as pd
import matplotlib
import json
from textwrap import wrap
from scipy import stats
import re
import sys


def standard_major(given_major):
    major = given_major.lower()
    if given_major == "macs":
        major = "computer science"
    elif given_major == "me":
        major = "mechanical engineering"
    elif given_major == "chemical engneering":
        major = "mechanical engineering"
    return major

def standard_country(given_country):
    country = given_country.lower()
    return country

def standard_languages(given_language):
    language = given_country.lower()
    return country
