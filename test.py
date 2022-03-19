import numpy
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa import stattools
from matplotlib import pyplot as plt
import pandas
import xlsxwriter

numpy.random.seed(1)

# test function 
def WNtest(data, lags):
    size = len(data)
    s = 0
    for lag in range(1, lags + 1):
        a1 = stats.pearsonr(data[lag:], data[:-lag])[0]
        a2 = stats.pearsonr(abs(data[lag:]), abs(data[:-lag]))[0]
        a3 = stats.pearsonr(data[lag:], abs(data[:-lag]))[0]
        a4 = stats.pearsonr(abs(data[lag:]), data[:-lag])[0]
        s = s + (a1**2 + a2**2 + a3**2 + a4**2)/(size - lag)
    statistics = size * (size + 2) * s
    pvalue = stats.chi2.sf(statistics, 4 * lags)
    return pvalue

# Ljung-Box test
def LBtest(data, lags):
    acf_resid, stat_resid, p_resid = stattools.acf(data, unbiased = True, fft=False, qstat = True)
    return p_resid[lags-1]
    
# Common parameters for all simulations
N = 100 # sample size
K = 5 # number of lags
z = numpy.random.normal(0, 1, N) # Gaussian white noise
w = numpy.random.laplace(0, 1, N) # Laplace white noise
# Gaussian white noise for log volatility
volNoise = numpy.random.normal(0, 1, N) 

allA = [0.1, 0.2, 0.3, 0.4, 0.5]

# simulation of the autoregression of order one
# with given noise and parameter
def ARsim(noise, a):
    ar = numpy.array([0])
    size = len(noise)
    for k in range(size):
        new = a * ar[k] + noise[k]
        ar = numpy.append(ar, new)
    return ar

# simulation of the moving average model
# with given noise and parameter
def MAsim(noise, a):
    return noise[:-1] + a * noise[1:]

# simulation of the stochastic volatility model
# with given noise for observations
# and independent Gaussian noise for volatility
def SVsim(noise, a):
    N = len(noise)
    v = ARsim(volNoise, a)[1:]
    x = numpy.exp(v) * noise
    return x

# simulation of the GARCH(1, 1) model
# with given noise for observations
# and given three parameters
def GARCHsim(noise, a, b, c):
    N = len(noise)
    x = numpy.array([0])
    v = numpy.array([0])
    for k in range(N):
        sigma = numpy.sqrt(a + b * x[k]**2 + c * v[k]**2)
        v = numpy.append(v, sigma)
        x = numpy.append(x, sigma*noise[k])
    return x[1:]

# AR(1) with Gaussian white noise
# Results of three tests:
# Ljung-Box for original values
# Ljung-Box for absolute values
# Our new test
ARGaussian = []
for a in allA:
    results = []
    ar = ARsim(z, a)
    results.append(LBtest(ar, K))
    results.append(LBtest(abs(ar), K))
    results.append(WNtest(ar, K))
    ARGaussian.append(results)
    
# AR(1) with Laplace white noise
# Results of three tests:
# Ljung-Box for original values
# Ljung-Box for absolute values
# Our new test
ARLaplace = []
for a in allA:
    results = []
    ar = ARsim(w, a)
    results.append(LBtest(ar, K))
    results.append(LBtest(abs(ar), K))
    results.append(WNtest(ar, K))
    ARLaplace.append(results)
    
# MA(1) with Gaussian white noise
# Results of three tests:
# Ljung-Box for original values
# Ljung-Box for absolute values
# Our new test
MAGaussian = []
for a in allA:
    results = []
    ma = MAsim(z, a)
    results.append(LBtest(ma, K))
    results.append(LBtest(abs(ma), K))
    results.append(WNtest(ma, K))
    MAGaussian.append(results)

# MA(1) with Laplace white noise
# Results of three tests:
# Ljung-Box for original values
# Ljung-Box for absolute values
# Our new test
MALaplace = []
for a in allA:
    results = []
    ma = MAsim(w, a)
    results.append(LBtest(ma, K))
    results.append(LBtest(abs(ma), K))
    results.append(WNtest(ma, K))
    MALaplace.append(results)

# SV with log volatility = AR(1) with independent Gaussian noise
# and Gaussian noise for observations
# Results of three tests:
# Ljung-Box for original values
# Ljung-Box for absolute values
# Our new test
SVGaussian = []
for a in allA:
    results = []
    sv = SVsim(z, a)
    results.append(LBtest(sv, K))
    results.append(LBtest(abs(sv), K))
    results.append(WNtest(sv, K))
    SVGaussian.append(results)

# SV with log volatility = AR(1) with independent Gaussian noise
# and Laplace noise for observations
# Results of three tests:
# Ljung-Box for original values
# Ljung-Box for absolute values
# Our new test
SVLaplace = []
for a in allA:
    results = []
    sv = SVsim(w, a)
    results.append(LBtest(sv, K))
    results.append(LBtest(abs(sv), K))
    results.append(WNtest(sv, K))
    SVLaplace.append(results)

# GARCH with Gaussian noise and equal parameters 
# Results of three tests:
# Ljung-Box for original values
# Ljung-Box for absolute values
# Our new test
GARCHGaussian = []
for a in allA:
    results = []
    garch = GARCHsim(z, a/3, a/3, a/3)
    results.append(LBtest(garch, K))
    results.append(LBtest(abs(garch), K))
    results.append(WNtest(garch, K))
    GARCHGaussian.append(results)

# GARCH with Laplace noise and equal parameters 
# Results of three tests:
# Ljung-Box for original values
# Ljung-Box for absolute values
# Our new test
GARCHLaplace = []
for a in allA:
    results = []
    garch = GARCHsim(w, a/3, a/3, a/3)
    results.append(LBtest(garch, K))
    results.append(LBtest(abs(garch), K))
    results.append(WNtest(garch, K))
    GARCHLaplace.append(results)

# Writing all results in Excel file, various pages
writer = pandas.ExcelWriter('test.xlsx', engine='xlsxwriter')
workbook = writer.book

df = pandas.DataFrame(ARGaussian)
df.to_excel(writer, 'AR-Gaussian', 2, 2)

df = pandas.DataFrame(ARLaplace)
df.to_excel(writer, 'AR-Laplace', 2, 2)

df = pandas.DataFrame(MAGaussian)
df.to_excel(writer, 'MA-Gaussian', 2, 2)

df = pandas.DataFrame(MALaplace)
df.to_excel(writer, 'MA-Laplace', 2, 2)

df = pandas.DataFrame(SVGaussian)
df.to_excel(writer, 'SV-Gaussian', 2, 2)

df = pandas.DataFrame(SVLaplace)
df.to_excel(writer, 'SV-Laplace', 2, 2)

df = pandas.DataFrame(GARCHGaussian)
df.to_excel(writer, 'GARCH-Gaussian', 2, 2)

df = pandas.DataFrame(GARCHLaplace)
df.to_excel(writer, 'GARCH-Laplace', 2, 2) 

writer.close()