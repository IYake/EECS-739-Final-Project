import matplotlib.pyplot as plt
import numpy as np

index = 0
x = []
with open('weights.txt') as f:
    lines = f.readlines()
    num = len(lines)
    x = list(range(num))
    
    true_b0 = float(lines[0].split()[0])
    true_b1 = float(lines[0].split()[1])
    b0 = [float(line.split()[0]) for line in lines[1:num]]
    b1 = [float(line.split()[1]) for line in lines[1:num]]

#b0 and b1 are single values
def f(x,b0,b1):
    return b0*x*x+x*b1

#b0 and b1 are arrays
def f_avg(x,b0,b1):
    avg = 0
    for i in range(len(b0)):
        if (abs(b0[i]) < 20):
            if (abs(b0[i]) > 0):
                avg += f(x,b0[i],b1[i])
    return avg/len(b0)
#
X = np.arange(0,20,0.1)

for i in range(len(b0)):
    if (abs(b0[i]) < 20 and b0[i] > 1):
        plt.plot(X,f(X,b0[i],b1[i]),'c')

plt.plot(X,f_avg(X,b0,b1),'p')
plt.plot(X,f(X,true_b0,true_b1),'r')

plt.title("True function vs Predictions in bag: ax^2+bx")

#mean squared error
#b0 and b1 are single values
def mean_squared_error(X,b0,b1):
    sum_ = 0
    for x in X:
        val = f(x,b0,b1)-f(x,true_b0,true_b1)
        sum_ += val*val
    return sum_ / len(X)

#b0 and b1 are arrays
def mean_squared_error_favg(X,b0,b1):
    sum_ = 0
    for x in X:
        val = f_avg(x,b0,b1)-f(x,true_b0,true_b1)
        sum_ += val*val
    return sum_ / len(X)

def avg_mean_squared_error(X,b0,b1):
    return np.average([mean_squared_error(X,b0[i],b1[i]) for i in range(len(b0))])

print(avg_mean_squared_error(X,b0,b1))
print(mean_squared_error_favg(X,b0,b1))

    








