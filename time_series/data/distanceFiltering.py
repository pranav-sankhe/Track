import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
# import pdb

readFileName = 'sorted_data/pos1_sorted.csv'
writeFileName = 'sorted_data/pos1_sorted_filtered.csv'

rows = []
with open(readFileName) as csvReader:
    reader = csv.reader(csvReader, delimiter=',')
    fields = reader.next()
    for row in reader:
        rows.append(row)

rows = np.array(rows)
distance = np.array(map(float, rows[:,1]))

filteredDistance = signal.medfilt(distance, kernel_size=11)
# plt.figure(1)
# plt.subplot(211),plt.plot(distance,'.')
# plt.subplot(212),plt.plot(filteredDistance,'.')
# plt.show()

# writing the filtered distance back to CSV file
rows[:,1] = filteredDistance
with open(writeFileName, 'w') as csvWriter:
	writer = csv.writer(csvWriter)
	writer.writerows(rows)

print("Filtered Data Saved in csv file")