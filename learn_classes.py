import matplotlib.pyplot as plt
import numpy as np


fig, ax = plt.subplots()


file_data = open("lear.txt")
colors = ['black', 'blue']
for i in range(2):
    x, y = [], []
    s = file_data.readline()
    while s.strip() !="":
        p = s.split()

        x.append(float(p[0]))
        y.append(float(p[1]))

        s = file_data.readline()  
    ax.scatter(x, y, color = colors[i],marker=".", s = 20, label = f"точки {i} класса")

file_data.close()


ax.grid(color='gray', linestyle='--', linewidth=0.5)
ax.axhline(0, color='black', linewidth=1)
ax.axvline(0, color='black', linewidth=1)
ax.legend()

plt.show()

