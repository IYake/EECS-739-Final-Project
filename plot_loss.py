
import matplotlib.pyplot as plt
import numpy as np

index = 0
x = []
with open('loss.txt') as f:
    lines = f.readlines()
    num = 1000
    x = list(range(num))
    y = [float(line.split()[0]) for line in lines[0:num]]

plt.plot(np.asarray(x),np.asarray(y))
plt.xlabel("iterations")
plt.ylabel("loss")

