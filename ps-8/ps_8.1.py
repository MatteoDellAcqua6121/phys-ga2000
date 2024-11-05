import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

#import data and devide it into two arrays: ages and answers
data = np.genfromtxt('survey.csv',
                     skip_header=1,
                     skip_footer=0,
                     dtype=np.float64,
                     delimiter=',')

ages=np.transpose(data)[0]
answers=np.transpose(data)[1]

#define the model and the likelyhood function
def p(x, b0, b1):
    return 1/(1.+np.exp(-(b0+b1*x)))

def likely(variables):
    b0,b1=variables
    P=p(ages, b0,b1)
    eps=1e-5
    return -np.sum((answers*np.log(P+eps)+(1-answers)*np.log(1-P+eps)))

#minimize and print the results
initial=[-10.,1.]
res = sp.optimize.minimize(likely, initial)
print(res)
print(np.sqrt(np.diagonal(res.hess_inv)))

#plot data vs model
m=1000
x0=np.linspace(np.min(data[:,0]),np.max(data[:,0]), m)

plt.plot(data[:,0],data[:,1], 'o', label='data')
plt.plot(x0, p(x0,*res.x), label='fit')
plt.xlabel('Age [years]')
plt.ylabel('p(x)')
plt.title('Logistic distribution')
plt.legend()
plt.savefig('logistic.png')
