import csv


file_path = './Data/Supercon_data.csv'

reader = csv.reader(open(file_path, 'r'))

compounds = []
temperatures = []

for row in reader:
   comp, temp = row
   compounds.append(comp)
   temperatures.append(comp)


print(compounds)
