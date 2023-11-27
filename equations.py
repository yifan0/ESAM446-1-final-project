import timesteppers
import finite
from scipy import sparse
import numpy as np

class KPZ:
    def __init__(self,c,spatial_order,domain,nu,lamda):
        self.t = 0
        self.iter = 0
        self.dt = None
        self.X = timesteppers.StateVector([c])
        grid_x,grid_y = domain.grids
        d2x = finite.DifferenceUniformGrid(2, spatial_order, grid_x, 0)
        d2y = finite.DifferenceUniformGrid(2, spatial_order, grid_y, 1)
        dx = finite.DifferenceUniformGrid(1, spatial_order, grid_x, 0)
        dy = finite.DifferenceUniformGrid(1, spatial_order, grid_y, 1)
        class Diffusionx:
            def __init__(self,c,nu,d2x):
                self.X = timesteppers.StateVector([c], axis=0)
                N = c.shape[0]
                self.M = sparse.eye(N, N)
                self.L = -(nu*d2x.matrix)
        class Diffusiony:
            def __init__(self,c,nu,d2y):
                self.X = timesteppers.StateVector([c], axis=1)
                N = c.shape[0]
                self.M = sparse.eye(N, N)
                self.L = -(nu*d2y.matrix)        
        diffx = Diffusionx(c,nu,d2x)
        diffy = Diffusiony(c,nu,d2y)
        self.ts_x = timesteppers.CrankNicolson(diffx,0)
        self.ts_y = timesteppers.CrankNicolson(diffy,1)
        
        class FE:
            def __init__(self,c,dx,dy,d2x,d2y,nu,lamda):
                self.X = timesteppers.StateVector([c])
                N = c.shape[0]
                def m(x,cc=0.5):
                    return (1-np.exp(-cc*np.square(x)))/cc
                def f(X):
#                     return lamda/2*(m(dx@X.data)+m(dy@X.data))
                    return lamda/2*(m(dx@X.data)+m(dy@X.data))+0.1*np.sqrt(12/0.04)*np.random.uniform(-0.5,0.5,size=(N,N))
                self.F = f
        self.ts_fe = timesteppers.ForwardEuler(FE(c,dx,dy,d2x,d2y,nu,lamda))
    def step(self,dt):
        self.ts_x.step(dt/2)
        self.ts_y.step(dt/2)
        self.ts_fe.step(dt)
        self.ts_y.step(dt/2)
        self.ts_x.step(dt/2)
        self.t += dt
        self.iter += 1
        
        
class Master:
    def __init__(self,c,spatial_order,domain,nu,lamda):
        self.t = 0
        self.iter = 0
        self.dt = None
        self.X = timesteppers.StateVector([c])
        grid_x,grid_y = domain.grids
        d2x = finite.DifferenceUniformGrid(2, spatial_order, grid_x, 0)
        d2y = finite.DifferenceUniformGrid(2, spatial_order, grid_y, 1)
        dx = finite.DifferenceUniformGrid(1, spatial_order, grid_x, 0)
        dy = finite.DifferenceUniformGrid(1, spatial_order, grid_y, 1)
        class Diffusionx:
            def __init__(self,c,nu,d2x):
                self.X = timesteppers.StateVector([c], axis=0)
                N = c.shape[0]
                self.M = sparse.eye(N, N)
                self.L = -(nu*d2x.matrix)
        class Diffusiony:
            def __init__(self,c,nu,d2y):
                self.X = timesteppers.StateVector([c], axis=1)
                N = c.shape[0]
                self.M = sparse.eye(N, N)
                self.L = -(nu*d2y.matrix)        
        diffx = Diffusionx(c,nu,d2x)
        diffy = Diffusiony(c,nu,d2y)
        self.ts_x = timesteppers.CrankNicolson(diffx,0)
        self.ts_y = timesteppers.CrankNicolson(diffy,1)
        
        class FE:
            def __init__(self,c,dx,dy,d2x,d2y,nu,lamda):
                self.X = timesteppers.StateVector([c])
                N = c.shape[0]
                def m(x,cc=0.5):
                    return (1-np.exp(-cc*np.square(x)))/cc
                def f(X):
#                     return lamda/2*(m(dx@X.data)+m(dy@X.data))
                    return lamda/2*(m(dx@X.data)+m(dy@X.data))+np.sqrt(3e-4+0.075*(m(dx@X.data)+m(dy@X.data)))*np.sqrt(12/0.04)*np.random.uniform(-0.5,0.5,size=(N,N))
                self.F = f
        self.ts_fe = timesteppers.ForwardEuler(FE(c,dx,dy,d2x,d2y,nu,lamda))
    def step(self,dt):
        self.ts_x.step(dt/2)
        self.ts_y.step(dt/2)
        self.ts_fe.step(dt)
        self.ts_y.step(dt/2)
        self.ts_x.step(dt/2)
        self.t += dt
        self.iter += 1
        