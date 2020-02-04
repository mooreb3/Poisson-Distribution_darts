# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 07:51:06 2019

@author: benji
"""
print('Benjamin Moore 17327505')
import matplotlib.pyplot as plt
import numpy as np
from numpy import math
import random
"part 1: Write Python code to plot the Poisson distribution, eqn(1), for hni = 1, 5, 10."
end=50
steps=50
n=np.linspace(0,end,steps) #defining the number of possible darts in a specific region

def poisson_dist(n_ave,n):
    f= ((n_ave**n)/(math.factorial(n)))*(np.exp(-n_ave))
    return f
#na is the expectation value
mean_manually=[1,5,10]
prob=[]
prob1=[]
prob2=[]
for i in range(len(n)):
    prob.append(poisson_dist(mean_manually[0],i))
    prob1.append(poisson_dist(mean_manually[1],i))
    prob2.append(poisson_dist(mean_manually[2],i))
    
plt.figure(1)    
plt.plot(n,prob)
plt.xlabel('n, number of darts in region')
plt.ylabel('probability value')
#plt.xlim(0,50)
plt.title('Poisson for mean equal to 1')
plt.show()

plt.figure(2)
plt.plot(n,prob1)
plt.xlabel('n, number of darts in region')
plt.ylabel('probability value')
#plt.xlim(0,50)
plt.title('Poisson for mean equal to 5')
plt.show()

plt.figure(3)
plt.plot(n,prob2)
plt.xlabel('n, number of darts in region')
plt.ylabel('probability value')
#plt.xlim(0,50)
plt.title('Poisson for mean equal to 10')
plt.show()    

"""Part2: P(n), nP(n), n^2P(n) sums"""

x=np.sum(prob)
x1=np.sum(prob1)
x2=np.sum(prob2) #summing up total probability
print ('sum of probs= ', x2)
ztot=0 #n^2P(n) sum
exp=0 #nP(n) sum
for i in range(steps):
    y=prob2[i]*i #probability for a point to have i darts multiplied by the number of darts
    z=prob2[i]*i**2 #probabilty multiplied by number of darts squared
    exp+=y #expected value can be plotted
    ztot+=z 

standarddeviation=np.sqrt(abs(exp**2-ztot))

print ('expected value of n=', exp)
print ('n^2P(n) sum=', ztot)
print ('standard deviation=', standarddeviation)





"""Part 3 and 4:
(i)Write a Python program to simulate the dart problem. Throw N = 50 darts at
random in one trial and initially set the number of regions to L = 100
(ii)Run ntrial = 5 “experiments” and determine H(n), the number of regions that have
n darts, and the mean number of darts per region, hni. 
(iii)Normalize H(n) to determine the probability distribution, eqn
(iv)Repeat the simulations for ntrial = 1 and ntrial = 10
(v)Now set L = 30 and N = 50 and repeat the simulation for ntrial = 1 and
ntrial = 100
(vi)What do you observe if the value for L becomes too small, e.g. L = 5


"""
N= 50            #number of darts         [50, 50, 50, 50, 50, 50]
L= 30    #number of regions       [100, 100, 100, 30, 30, 5] 
ntrial= 10        #number of 'experiments' [1, 5, 10, 1, 100, 1000]

#could do this with sumcount=[] and then appending values
sumc=[0]*N
averagec=[0]*N
B=[0]*L
tcount=[0]*N

class experiments():
    #simulating the dart throwing by randomly applying a dart to the regions
    def throw(self,B):
        for i in range(L):
            B[i]=0
        for i in range(N):
            a=random.randint(0,L-1)
            B[a] += 1
        return B
            
    #counting the number of darts per region using same code again/number of regions with n darts
    def dpr(self,B,tcount):
        self.throw(B)
        for i in range(N):
            tcount[i]=B.count(i)
        return tcount

    #Getting average count over all the number of iterations
    def iterate(self,ntrial,B,tcount):
        for i in range(ntrial):
            self.dpr(B,tcount)
            for i in range(N):
                sumc[i]+=tcount[i]/ntrial
        for i in range(N):
            averagec[i]=sumc[i]
        return averagec

experiments().iterate(ntrial,B,tcount)

probability_dist=[0]*N
for i in range(N):
    probability_dist[i]=averagec[i]/L

newmeans=[]
for i in range(N):
    x=i*probability_dist[i]
    newmeans.append(x)
expval=np.sum(newmeans)
print(expval)
    
probability1=[]
for i in range(len(n)):
    probability1.append(poisson_dist(expval,i))

plt.plot(probability_dist, label=' Result of randomly picking region')
plt.ylabel("probability")
plt.xlabel("n, number of darts in region")
plt.title("Poisson distribution  Vs. a random sampling for 50 darts, into 30 regions, averaged over 1 experiments")
plt.xlim(0,20)
plt.plot(n,probability1, label='Poisson Distribution')
plt.legend(loc='upper right')
plt.show()