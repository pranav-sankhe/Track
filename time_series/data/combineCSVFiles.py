"""
    Run using python2 only
"""
import csv
import numpy as np
import random

# filenames = ['sorted_data/31-05_2_sorted.csv', 'sorted_data/31-05_3_sorted.csv', 'sorted_data/31-05_4_sorted.csv', 'sorted_data/pos1_sorted.csv']
filenames = ['sorted_data/31-05_2_sorted_filtered.csv', 'sorted_data/31-05_3_sorted_filtered.csv', 'sorted_data/pos1_sorted_filtered.csv', 'sorted_data/31-05_4_sorted_filtered.csv']
# filenames = ['sorted_data/31-05_2_sorted.csv', 'sorted_data/pos1_sorted.csv', 'sorted_data/31-05_4_sorted.csv', 'sorted_data/31-05_3_sorted.csv']
# filenames = ['sorted_data/pos1_sorted.csv', 'sorted_data/31-05_4_sorted.csv', 'sorted_data/31-05_3_sorted.csv', 'sorted_data/31-05_2_sorted.csv']

resultFile = 'sorted_data/result_filtered.csv'
rows = []

for i in range(4):
    with open(filenames[i]) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        fields = csvreader.next()
        for row in csvreader:
            rows.append(row)

rows = np.array(rows)
# writing to csv file
with open(resultFile, 'w') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)
     
    # writing the fields
    csvwriter.writerow(fields)
     
    # writing the data rows
    csvwriter.writerows(rows)