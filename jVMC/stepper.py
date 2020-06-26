import jax
import jax.numpy as jnp
import numpy as np

class Euler:

    def __init__(self, timeStep=1e-3):

        self.dt = timeStep


    def step(self, t, f, yInitial, rhsArgs=None):

        dy = f(yInitial, t, rhsArgs)

        return yInitial + self.dt * dy, self.dt

# end class Euler


class AdaptiveHeun:

    def __init__(self, timeStep=1e-3, tol=1e-8, maxStep=1.):
        self.dt = timeStep
        self.tolerance = tol
        self.maxStep = maxStep


    def step(self, t, f, yInitial, normFunction, rhsArgs=None):

        fe = 0.5

        dt = self.dt

        while fe < 1.:
        
            y = yInitial.copy()
            k0 = f(y, t, rhsArgs)
            y += dt * k0
            k1 = f(y, t + dt, rhsArgs)
            dy0 = 0.5 * dt * (k0 + k1)
            
            # now with half step size
            y -= 0.5 * dt * k0
            k10 = f(y, t + 0.5 * dt, rhsArgs)
            dy1 = 0.25 * dt * (k0 + k10)
            y = yInitial + dy1
            k01 = f(y, t + 0.5 * dt, rhsArgs)
            y += 0.5 * dt * k01
            k11 = f(y, t + dt, rhsArgs)
            dy1 += 0.25 * dt * (k01 + k11)

            # compute deviation
            updateDiff = normFunction(dy1 - dy0)
            fe = self.tolerance / updateDiff

            if 0.2>0.9*fe**0.33333:
                tmp = 0.2
            else:
                tmp = 0.9*fe**0.33333
            if tmp > 2.:
                tmp = 2.

            realDt = dt
            dt *= tmp

            if dt > self.maxStep:
                dt = self.maxStep

        # end while

        self.dt = dt

        return yInitial + dy1, realDt


#def f(y,t,args):
#    return args['mat'].dot(y)
#
#def norm(y):
#    return jnp.real(jnp.conjugate(y).dot(y))
#
#stepper = AdaptiveHeun()
#rhsArgs={}
#rhsArgs['mat'] = jnp.array(np.random.rand(4,4) + 1.j * np.random.rand(4,4))
#
#y0 = jnp.array(np.random.rand(4))
#y = y0.copy()
#t=0
#diffs=[]
#for k in range(100):
#    y, dt = stepper.step(t,f,y,normFunction=norm,rhsArgs=rhsArgs)
#    t+=dt
#    yExact = jax.scipy.linalg.expm(t * rhsArgs['mat']).dot(y0)
#    diff = y - yExact
#    diffs.append(norm(diff))
#
#print(diffs)
#print(np.max(np.array(diffs)))
