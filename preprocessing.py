from mendeleev import element
import numpy as np

import csv
import re

NUM_ELEMENTS = 100

file_path = './Data/Supercon_data.csv'



def compound_to_array(compound_name, NUM_ELEMENTS):
    print(compound_name)
    one_hot = np.eye(NUM_ELEMENTS)
    data = []
    for s in compound_name:
        if s.isnumeric() or '.' in s:
            data[-1].append(float(s))
        else:
            i = element(s).atomic_number
            data.append(list(one_hot[i]))
    return np.array(data)

def load_data(path, NUM_ELEMENTS = NUM_ELEMENTS):

    reader = csv.reader(open(file_path, 'r'))

    compound_names = []
    temperatures = []

    for row in reader:
       comp, temp = row
       compound_names.append(comp)
       temperatures.append(temp)



    inputs = []

    for i,comp in enumerate(compound_names):
        if 'OY' in comp:
            #print("NAME: ", comp, " " * (40 - len(comp)) + "TEMP: ", temperatures[i])
            continue
        split = re.findall('[\d.]+|[A-Z][a-z]*', comp)

        inputs.append(compound_to_array(split, NUM_ELEMENTS))

    return np.array(inputs), np.array(labels)

load_data(file_path)
