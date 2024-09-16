#パスの追加
import os, sys
from pathlib import Path

#各種ライブラリの読み込み
import pandas as pd
import numpy as np
import tqdm
import re
import itertools

import gensim
from gensim.models import KeyedVectors

from predication import create_categorization_int
from ir_ import average_precision, average_precision_for_check

