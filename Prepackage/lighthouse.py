# apply.c     "LIGHTHOUSE" NESTED SAMPLING APPLICATION
# (GNU General Public License software, (C) Sivia and Skilling 2006)
# translated to Python by Issac Trotts in 2007
#
#              u=0                                 u=1
#               -------------------------------------
#          y=2 |:::::::::::::::::::::::::::::::::::::| v=1
#              |::::::::::::::::::::::LIGHT::::::::::|
#         north|::::::::::::::::::::::HOUSE::::::::::|
#              |:::::::::::::::::::::::::::::::::::::|
#              |:::::::::::::::::::::::::::::::::::::|
#          y=0 |:::::::::::::::::::::::::::::::::::::| v=0
# --*--------------*----*--------*-**--**--*-*-------------*--------
#             x=-2          coastline -->east      x=2
# Problem:
#  Lighthouse at (x,y) emitted n flashes observed at D[.] on coast.
# Inputs:
#  Prior(u)    is uniform (=1) over (0,1), mapped to x = 4*u - 2; and
#  Prior(v)    is uniform (=1) over (0,1), mapped to y = 2*v; so that
#  Position    is 2-dimensional -2 < x < 2, 0 < y < 2 with flat prior
#  Likelihood  is L(x,y) = PRODUCT[k] (y/pi) / ((D[k] - x)^2 + y^2)
# Outputs:
#  Evidence    is Z = INTEGRAL L(x,y) Prior(x,y) dxdy
#  Posterior   is P(x,y) = L(x,y) / Z estimating lighthouse position
#  Information is H = INTEGRAL P(x,y) log(P(x,y)/Prior(x,y)) dxdy

from math import *
import random
from mininest import nested_sampling
import numpy as np
import matplotlib.pyplot as plt
import plot

n=100                   # number of objects
max_iter = 1000        # number of iterations

class Object:
    def __init__(self):
        self.u=None     # Uniform-prior controlling parameter for x
        self.v=None     # Uniform-prior controlling parameter for y
        self.x=None     # Geographical easterly position of lighthouse
        self.y=None     # Geographical northerly position of lighthouse
        self.logL=None  # logLikelihood = ln Prob(data | position)
        self.logWt=None # log(Weight), adding to SUM(Wt) = Evidence Z

uniform = random.random
N = 500
R = np.array([[3.4,2.75],[2.75,1.5]])
D = np.random.multivariate_normal((1.2,0.5), R, size=N)

#C = np.array([[1,0],[0,1]])
#C = np.cov(np.stack((D[:,0],D[:,1]), axis=0))

#print(C)

# 250-D iid standard normal log-likelihood

def logLhood(x,y,x_corr,y_corr,xy_corr):
	C = np.array([[x_corr, xy_corr],[xy_corr, y_corr]])
	print(np.linalg.det(C))
	ndim = 2
	Cinv = np.linalg.inv(C)
	lnorm = -0.5 * (np.log(2 * np.pi) * ndim + np.log(np.linalg.det(C)))  # ln(normalization)
	pos = np.array([x,y])
	
	N = len(D)
	logL = 0.0
	for k in range(N):
		logL += -0.5 * np.dot(((pos-D[k]).T), np.dot(Cinv, (pos-D[k]))) + lnorm
	return logL

def sample_from_prior():
	Obj = Object()
	Obj.u = random.random()
	Obj.v = random.random()
	Obj.xc = random.random()
	Obj.yc = random.random()
	Obj.xyc = random.random()
	Obj.x = 12 * Obj.u - 3.0
	Obj.y = 12 * Obj.v - 3.0
	Obj.x_corr = 3 * Obj.xc
	Obj.y_corr = 3 * Obj.yc
	Obj.xy_corr = 3 * Obj.xyc - 1.5
	Obj.logL = logLhood(Obj.x, Obj.y, Obj.x_corr, Obj.y_corr, Obj.xy_corr)
	return Obj

# Note that unlike the C version, this function returns an 
# updated version of Obj rather than changing the original.
def explore(   # Evolve object within likelihood constraint
    Obj,       # Object being evolved
    logLstar): # Likelihood constraint L > Lstar
	ret = Object()
	ret.__dict__ = Obj.__dict__.copy()
	step = 0.1
	accept = 0
	reject = 0
	Try = Object()
	
	for m in range(20):
		Try.u = ret.u + step * (2.*uniform() - 1.)
		Try.v = ret.v + step * (2.*uniform() - 1.)
		Try.xc = ret.xc + step * (2.*uniform() - 1.)
		Try.yc = ret.yc + step * (2.*uniform() - 1.)
		Try.xyc = ret.xyc + step * (2.*uniform() - 1.)
		Try.u -= floor(Try.u)
		Try.v -= floor(Try.v)
		Try.xc -= floor(Try.xc)
		Try.yc -= floor(Try.yc)
		Try.xyc -= floor(Try.xyc)
		Try.x = 12 * Try.u - 3
		Try.y = 12 * Try.v - 3
		Try.x_corr = 3 * Try.xc
		Try.y_corr = 3 * Try.yc
		Try.xy_corr = 3 * Try.xyc - 1.5
		Try.logL = logLhood(Try.x, Try.y, Try.x_corr, Try.y_corr, Try.xy_corr);  # trial likelihood value
		
		if Try.logL > logLstar:
			ret.__dict__ = Try.__dict__.copy()
			accept+=1
		else:
			reject+=1

        # Refine step-size to let acceptance ratio converge around 50%
		if( accept > reject ):   step *= exp(1.0 / accept)
		if( accept < reject ):   step /= exp(1.0 / reject)
		return ret

def process_results(results):
	(x,xx) = (0.0, 0.0)
	(y,yy) = (0.0, 0.0)
	(xc, yc, xyc) = (0.0, 0.0, 0.0)
	ni = results['num_iterations']
	samples = results['samples']
	logZ = results['logZ']

	samplexlist = []
	sampleylist = []

	for i in range(ni):
		samplexlist.append(samples[i].x)
		sampleylist.append(samples[i].y)
		w = exp(samples[i].logWt - logZ); # Proportional weight
		x  += w * samples[i].x
		#print(x)
		xx += w * samples[i].x * samples[i].x
		y  += w * samples[i].y
		yy += w * samples[i].y * samples[i].y

		xc += samples[i].xc

	logZ_sdev = results['logZ_sdev']
	H = results['info_nats']
	H_sdev = results['info_sdev']
	print("# iterates: %i"%ni)
	print("Evidence: ln(Z) = %g +- %g"%(logZ,logZ_sdev))
	print("Information: H  = %g nats = %g bits"%(H,H/log(2.0)))
	print("mean(x) = %9.4f, stddev(x) = %9.4f"%(x, sqrt(xx-x*x)));
	print("mean(y) = %9.4f, stddev(y) = %9.4f"%(y, sqrt(yy-y*y)));

	return x, y, sqrt(xx-x*x), sqrt(yy-y*y), samplexlist, sampleylist

if __name__ == "__main__":
	results = nested_sampling(n, max_iter, sample_from_prior, explore)
	meanx, meany, stdx, stdy, samplexlist, sampleylist = process_results(results)
	
	#c_matrix = np.array([[stdx**2, cov], [cov, stdy**2]])
	#print(c_matrix)

	fig, ax = plt.subplots()
	plot.get_cov_ellipse(c_matrix, [meanx,meany], 1, ax, 'red')
	ax.scatter(samplexlist, sampleylist, color='limegreen', marker='x', s=2)

	sample_cov = np.cov(np.stack((samplexlist, sampleylist), axis=0))
	plot.get_cov_ellipse(C, [meanx,meany], 3, ax, 'black')
	ax.scatter(D[:,0], D[:,1], s=2)

	plt.show()