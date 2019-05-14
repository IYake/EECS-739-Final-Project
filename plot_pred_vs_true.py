#plot all the predictive models together with the true one
#also plot the combination of those predictive models

import numpy as np
import matplotlib.pyplot as plt

def f(x,b0,b1):
    return b0*np.exp(x*b1)

X = np.arange(8,8.001,0.00000001)

plt.plot(X,f(X,2.5411,0.2595),'r')
#plt.plot(X,f(X,2.540734,0.259518),'b')

plt.plot(X,f(X,2.541393,0.259484),'c')
plt.plot(X,f(X,2.541272,0.259486),'c')
plt.plot(X,f(X,2.540757,0.259519),'c')

plt.title("True function vs Predictions with diff step sizes")


