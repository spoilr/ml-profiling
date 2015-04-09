# from pymc import *
# import numpy as np

# # __all__ = [
# #     'OtherKnowledge',
# #     'RecruitNetGroup',
# #     'ClaimResp',
# #     'HighValueCivilian']

# # disaster_model = np.transpose(np.matrix([[1,1,0,1],
# # 							 [1,1,1,1],
# # 						 	 [0,1,0,1],
# # 							 [0,0,1,2],
# # 							 [1,1,1,2]]))

# # disaster_model = np.array([1,1,0,0,1,
# # 							1,1,1,0,1,
# # 							0,1,0,1,1,
# # 							1,1,1,2,2])


# from pymc.examples import disaster_model

# M = MCMC(disaster_model)
# M.sample(iter=10000, burn=1000, thin=10)
# M.trace('switchpoint')[:]



from pymc import rweibull, Uniform, Weibull


alpha = 3
beta = 5
N = 100
dataset = rweibull(alpha, beta, N)

print dataset

"""
Now we create a pymc model that defines the likelihood
of the data set and prior assumptions about the value
of the parameters.
"""
a = Uniform('a', lower=0, upper=10, value=5, doc='Weibull alpha parameter')
b = Uniform('b', lower=0, upper=10, value=5, doc='Weibull beta parameter')
like = Weibull('like', alpha=a, beta=b, value=dataset, observed=True)

if __name__ == '__main__':

    from pymc import MCMC, Matplot

    # Sample the parameters a and b and analyze the results
    M = MCMC([a, b, like])
    M.sample(10000, 5000)
    # Matplot.plot(M)