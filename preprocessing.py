from mendeleev import element
import numpy as np

import sys
import csv
import re

def compound_to_array(compound_name, NUM_ELEMENTS):
    one_hot = np.eye(NUM_ELEMENTS)
    data = []

    i = 0
    while i < len(compound_name):
        s = compound_name[i]
        if i+1 == len(compound_name):
            n = '1' #if at end then number is 1
        else:
            n = compound_name[i+1]

        if n.isnumeric() or '.' in n:
            N = float(n)
            i += 1
        else:
            N = 1


        if s == 'T':
            return np.array([])
        if s == 'D':
            return np.array([])
            # i+=1
            # data.append(np.append(one_hot[0],N))
            # continue

        try:
            ele = element(s)
        except:
            print("FAILED")
            print(s)
            print(compound_name)
            print()
            sys.exit("Unknown Element")

        _id = ele.atomic_number
        other = [N,ele.atomic_weight]
        other.append(ele.atomic_volume)
        other.append(ele.atomic_radius)
        other.append(ele.charge)
        other.append(ele.ionic_radius)
        other.append(ele.group_id)
        other.append(ele.electron_affinity)
        other.append(ele.en_pauling)
        other.append(ele.en_ghosh)
        other.append(ele.en_allen)
        other.append(ele.covalent_radius_bragg)
        other.append(ele.heat_of_formation)

        comp_vect = np.append(one_hot[id],np.array(other))
        data.append(comp_vect)

        i += 1
    return np.array(data)

def split_compound_string(comp):
    return re.findall('[\d.]+|[A-Z][a-z]*', comp)

def load_data(file_path, NUM_ELEMENTS=100):

    reader = csv.reader(open(file_path, 'r'))

    compound_names = []
    temperatures = []

    for row in reader:
        comp, temp = row
        if temp == 'Tc': #first row
            continue
        compound_names.append(comp)
        temperatures.append(float(temp))

    #compound_names = compound_names[:100]

    inputs = []
    size = len(compound_names)
    print("DATA SIZE: ", size)
    BAR = 50
    for i,comp in enumerate(compound_names):
        prog = int(i/size * BAR)
        print("LOADING |" + '#'* prog + ' '*(BAR-prog) + "|", str(i) + "/" + str(size) + "   ", end='\r')
        if 'OY' in comp:
            continue
        if '+' in comp:
            continue

        split = split_compound_string(comp)

        arr = compound_to_array(split, NUM_ELEMENTS)
        if not len(arr) == 0:
            inputs.append(arr)

    return np.array(inputs), np.array(temperatures)
