"""

"""

# Imports - standard
import os
import numpy as np
import pandas as pd

# Imports - custom
import sys
sys.path.append("code")
from paths import PROJECT_PATH
from utils import get_start_time, print_time_elapsed


def main():

    # display progress
    t_start = get_start_time()

    # identify / create directories
    dir_output = f"{PROJECT_PATH}/<path/to/output>"
    if not os.path.exists(dir_output): 
        os.makedirs(f"{dir_output}")




    # display progress
    print(f"\n\nTotal analysis time:")
    print_time_elapsed(t_start)


if __name__ == "__main__":
    main()
