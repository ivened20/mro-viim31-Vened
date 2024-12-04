import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

file_data = open("data.txt")
k_class = int(file_data.readline())
k_points_class = [int(i) for i in file_data.readline().split(" ")]
k_izm = int(file_data.readline())

#Черчение прямоугольников
fig, axes = plt.subplots(1, 2)
fl = True
for i in range(k_class):
      s_ranges = []
      min_ranges = []
      max_ranges = []
      s_ranges = [int(i) for i in file_data.readline().split(" ")]
      min_ranges = s_ranges[:int(len(s_ranges)/2)]
      max_ranges = s_ranges[int(len(s_ranges)/2):]
      width = max_ranges[0] - min_ranges[0]
      height = max_ranges[1] - min_ranges[1]
      if fl:
            rect1 = Rectangle((min_ranges[0], min_ranges[1]), width, height, facecolor = 'grey', label="class")
            rect2 = Rectangle((min_ranges[0], min_ranges[1]), width, height, facecolor = 'grey', label="class")
            fl = False
      else:
           rect1 = Rectangle((min_ranges[0], min_ranges[1]), width, height, facecolor = 'grey')
           rect2 = Rectangle((min_ranges[0], min_ranges[1]), width, height, facecolor = 'grey')
      axes[1].add_patch(rect1)
      axes[0].add_patch(rect2)
file_data.close()

iso = open("iso_filtered_pts.txt")
colors = ['green', 'blue', 'orange']
i_col = 0
s = iso.readline()
while s !="":
      x = []
      y = []
      pts = []
      while s.strip() != "":
          pts.append(s) 
          s = iso.readline() 
      for i in pts:
            coord = i.split()
            x.append(float(coord[0]))
            y.append(float(coord[1]))
      axes[0].scatter(x, y, color=colors[i_col % 3],marker=".", s=10, label="iso_points")
      s = iso.readline()
      i_col += 1
iso.close()



#Изображение точек
file_points = open("file_for_checking.txt")
x = []
y = []
points_count = sum(k_points_class)

for i in range(points_count):
      points = [float(i) for i in file_points.readline().split(" ") if i!="\n"]
      x.append([points[0]])
      y.append([points[1]])
file_points.close()
axes[1].scatter(x, y, color = "red", marker=".", s=6, label="points")


# Хо-Кашьяп
file_lines = open("kashyap_lines.txt")
params=[i for i in file_lines.readlines()]
file_lines.close()

            
            
# Создаем массив значений для оси x1
lim_line = open("data.txt")
for i in range(3):
      lim_line.readline()

atributes = [i for i in lim_line.readlines()]
ii = -1
for h in range(k_class-1):
      for hh in range(h+1,k_class):
            ii += 1
            x_ran=[]
                 
            for i in range(0, k_izm*k_izm, k_izm):
                  x_ran.append(float(atributes[h].split()[i]))
                  x_ran.append(float(atributes[hh].split()[i]))
                 
            line_params=[float(huha) for huha in params[ii].split()]
            x1 = np.linspace(min(x_ran), max(x_ran)) 
            # Рассчитываем значения x2 по уравнению x2 = ax1 + b
            x2 = (line_params[0]/-(line_params[1])) * x1 + line_params[2]/-(line_params[1])

            # Осуществляем построение графика
            axes[0].plot(x1, x2, label='(' + str(line_params[0])+')x1 + (' + str(line_params[1]) + ')x2 + (' + str(line_params[2]) + ') = 0')
lim_line.close()


# Кластеризация
clasters = open("k_means.txt")
gg=clasters.readline()
xcoord = []
ycoord = []
for i in range(int(gg)):
      coord = clasters.readline().split()
      xcoord.append(float(coord[0]))
      ycoord.append(float(coord[1]))
axes[1].scatter(xcoord, ycoord, color="b",marker="+", s=10, label="prosei")

xcoord = []
ycoord = []
for i in range(int(gg)):
      coord = clasters.readline().split()
      xcoord.append(float(coord[0]))
      ycoord.append(float(coord[1]))
axes[1].scatter(xcoord, ycoord, color="k",marker="+", s=10, label="maximinizir")

clasters.close()

forel = open("forel_centroids.txt")
xcoord = []
ycoord = []
r=int(forel.readline())
cts = forel.readlines()
for i in cts:
      coord = i.split()
      xcoord.append(float(coord[0]))
      ycoord.append(float(coord[1]))
axes[1].scatter(xcoord, ycoord, color="m",marker="*", s=10, label="forel")
fl = True
for i in range(len(cts)):
    if fl:
      circle = plt.Circle((xcoord[i], ycoord[i]), r, fill=False, color='black', label='Окружность', alpha=0.5)
      fl = False
    else:
      circle = plt.Circle((xcoord[i], ycoord[i]), r, fill=False, color='black', alpha=0.5)
    axes[1].add_patch(circle)
forel.close()

iso = open("iso.txt")
xcoord = []
ycoord = []
pts = iso.readlines()
for i in pts:
      coord = i.split()
      xcoord.append(float(coord[0]))
      ycoord.append(float(coord[1]))
axes[0].scatter(xcoord, ycoord, color="black",marker="s", s=10, label="iso_centers")
iso.close()

for ax in axes.flat:  # Итерация по всем осям
    ax.grid(color='gray', linestyle='--', linewidth=0.5)
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='black', linewidth=1)
    #ax.legend()

plt.show()
plt.tight_layout()

