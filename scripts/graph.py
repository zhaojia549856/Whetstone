import re
import sys
from os import getcwd
import matplotlib.pyplot as plt
import numpy as np
import glob

regex = re.compile(r"X= ([0-9.]*) ([0-9.]*) predicted: ([0-9]) real: ([0-9])")

if len(sys.argv) != 2:
	print("usage: python graph.py <filename>")
	exit(1)

currect = [[], []]
incurrect = [[], []]

with open(sys.argv[1], "r") as f:
	for line in f:
		result = regex.search(line)
		tmp = (float(result.group(1)), float(result.group(2)))
		if result:
			if int(result.group(4)) == 1:
				currect[0].append(tmp[0])
				currect[1].append(tmp[1])
			else:
				incurrect[0].append(tmp[0])
				incurrect[1].append(tmp[1])

x = np.linspace(0, 1, 10000)
plt.plot(currect[0], currect[1], '.', label="Currect guess")
plt.plot(incurrect[0], incurrect[1], '.', label="Incurrect guess")
plt.plot(x, 3*x-1, '-', label="y=3x-1")
plt.legend(loc='upper right', fancybox=True)
plt.show()