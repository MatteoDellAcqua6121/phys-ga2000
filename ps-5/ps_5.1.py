import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

#Copy the central diff function from the jupiter notebook of the class
def diff_central(func=None, x=None, dx=None):
    return((func(x + 0.5 * dx) - func(x - 0.5 * dx)) / dx)

#define the relevant functions: f; g its analytical derivative; f0,f1,f2 the factor appearing in his decomposition
def f(x):
    return 1.+0.5*np.tanh(2*x)

def g(x):
    return 1.-np.square(np.tanh(2*x))

def f0_jax(x):
    return 2*x

def f1_jax(x):
    return jnp.tanh(x)

def f2_jax(x):
    return 1.+0.5*x

def f_jax(x):
    return f2_jax(f1_jax(f0_jax(x)))

#prepare the plot
m=1000
x=np.linspace(-2.,2., m)
dfdx = diff_central(func=f, x=x, dx=0.5 * 1.e-5)
y=np.array([g(xi) for xi in x])

#run the jax autodiff
dv_jax = jax.grad(f_jax)
#  dv = dv_jax(x) <- this will fail because it expects a scalar
dv = jax.vmap(dv_jax)(x)  # but you can map the values into the function this way

#plot everything
plt.figure(figsize=(8, 6))
plt.plot(x, dv, label='Jax autodiff')
plt.plot(x, dfdx, linestyle='--',label='Central difference')
plt.plot(x, y, ':', label='Analytcal')
plt.title(r'$\frac{d}{dx}(1.+\operatorname{tanh}(2x)/2$')
plt.xlabel('x [au]')
plt.ylabel('y [au]')    
plt.legend()
plt.savefig('Derivative.png')