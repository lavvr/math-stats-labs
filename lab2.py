import numpy as np
import scipy.stats as sps
from abc import ABC, abstractmethod
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from math import log2
class TrustedIntervals(ABC):

    @abstractmethod
    def __init__(self, beta=0.95):
        alpha = (1 - beta) / 2
    
    @abstractmethod
    def buildIntervals(self):
        pass

    @abstractmethod
    def print(self):
        pass

class Gauss(TrustedIntervals):
    def __init__(self, mean=3, dev=np.sqrt(10), n=500, beta = 0.95):
        self.mean = mean
        self.dev = dev
        self.n = n
        self.alpha = (1 - beta) / 2
        
    def buildIntervals(self):
        np.random.seed(42)

        self.gauss_sample = sps.norm(self.mean, self.dev).rvs(self.n)
        mean = np.mean(self.gauss_sample)
        var = np.var(self.gauss_sample, ddof=1)
       
        #должно быть просто alpha, но в таком случае не получается
        self.left_side_mean = mean - sps.norm.ppf(1 - self.alpha) * self.dev / np.sqrt(self.n)
        self.right_side_mean = mean + sps.norm.ppf(1 - self.alpha) * self.dev / np.sqrt(self.n)
       
        self.right_side_var = (self.n - 1) * var / sps.chi2.ppf(self.alpha, self.n - 1)
        self.left_side_var =  (self.n - 1) * var / sps.chi2.ppf(1 - self.alpha, self.n - 1)

        self.groupingValues()

    def tukyMethod(self):
        vrt = sorted(self.gauss_sample)
        lq = np.percentile(self.gauss_sample, 25)
        uq = np.percentile(self.gauss_sample, 75)
        iqr = uq - lq

        xl = max(vrt[0], lq - 1.5 * iqr)
        xr = min(vrt[-1], uq + 1.5 * iqr)

        return [num for num in self.gauss_sample if xl <= num <= xr]

    def groupingValues(self):

        wout_emmisions = self.tukyMethod()
        k = 8#int(log2(len(self.gauss_sample)) + 1)

        x1 = np.min(wout_emmisions)
        xn = np.max(wout_emmisions)
        z0 = np.floor(x1)
        zk = np.floor(xn)
    
        cur_len = zk - z0
        check = cur_len % k

        if check != 0:
            delta = k - check
            zk += delta

        edges = np.linspace(z0, zk, num=k + 1)

        hist, _ = np.histogram(wout_emmisions, bins=edges)
        vi = (edges[:-1] + edges[1:]) / 2
        grouped_sample = np.repeat(vi, hist)
        n = len(grouped_sample)

        mean = np.mean(grouped_sample)
        var = np.var(grouped_sample, ddof = 1)
    

        t_val = sps.t.ppf(1 - self.alpha, df=k-1)
        self.t_left =  mean - t_val * np.sqrt(var / (k - 1))
        self.t_right =  mean + t_val * np.sqrt(var / (k - 1))

        chi2_left = sps.chi2.ppf(1 - self.alpha, df=k-1)
        chi2_right = sps.chi2.ppf(self.alpha, df=k-1)
        self.var_left = (k - 1) * var / chi2_left
        self.var_right = (k - 1) * var / chi2_right


    
    def print(self):

        print("Gauss trusted interval for mean:", end = ' ')
        print(f"({round(self.left_side_mean,3)}, {round(self.right_side_mean,3)})")
        print("Gauss trusted interval for dispersion:", end = ' ')
        print(f"({round(self.left_side_var,3)}, {round(self.right_side_var,3)})")
        

        print("\nGauss trusted interval for grouped mean")
        print(f"({round(self.t_left,3)}, {round(self.t_right,3)})")
        print("Gauss trusted interval for grouped dispersion")
        print(f"({round(self.var_left,3)}, {round(self.var_right,3)})")


        

class Bernulli(TrustedIntervals):

    def __init__(self, n=500, p=1/3,  beta=0.95, k=1):
        self.p = p
        self.n = n
        self.k = k
        self.alpha = (1-beta) / 2
    
    def buildIntervals(self):

        np.random.seed(42)
        self.bernulli_sample = sps.binom(self.k, self.p).rvs(self.n)
        m = np.sum(self.bernulli_sample)
        p = m/self.n
        t_alpha = sps.norm.ppf(1 - self.alpha)

        if self.n >= 100:
            
            
            coeff = np.sqrt(p * (1 - p) / self.n)
            self.left_side = p - t_alpha * coeff
            self.right_side = p + t_alpha * coeff

        elif self.n > 10:
            coeff1 = (t_alpha ** 2) / (2*self.n)  
            coeff2 = np.sqrt(p * (1 - p) / self.n + (t_alpha ** 2) / (4 * self.n ** 2)) 
            self.left_side = self.n *( p + coeff1 - t_alpha * coeff2) / (self.n + t_alpha ** 2)
            self.right_side = self.n *( p + coeff1 + t_alpha * coeff2) / (self.n + t_alpha ** 2)
     
    def print(self):
        
        if self.n >= 100:
            print("\nBernully trusted for p(more than 100 experiments)")
            print(f"({round(self.left_side,3)}, {round(self.right_side,3)})")

        elif self.n > 10:
            print("\nBernully trusted for p(between 10 and 100 experiments)")
            print(f"({round(self.left_side,3)}, {round(self.right_side,3)})")


class Puasson(TrustedIntervals):

    def __init__(self, gamma=2, n=500, beta = 0.95):
        self.gamma = gamma
        self.n = n
        self.alpha = (1 - beta) / 2

    def buildIntervals(self):

        np.random.seed(42)
        self.puasson_sample = sps.poisson(mu = self.gamma).rvs(self.n)
        gamma = np.mean(self.puasson_sample)

        t_alpha = sps.norm.ppf(1 - self.alpha )
        self.left_side = gamma - t_alpha * np.sqrt(gamma / self.n)
        self.right_side = gamma + t_alpha * np.sqrt(gamma / self.n)

    def print(self):
        print(f"\nPuasson trusted interval for lambda = {self.gamma}")
        print(f"({round(self.left_side,3)}, {round(self.right_side,3)})")


class Exponential(TrustedIntervals):

    def __init__(self, n=500, sigma=3, beta=0.95):
        self.n = n
        self.sigma = sigma
        self.alpha = (1 - beta) / 2

    def buildIntervals(self):

        np.random.seed(42)
        self.exponential_sample = sps.expon(scale=1/self.sigma).rvs(self.n)
        mean = np.mean(self.exponential_sample)
        t_alpha = sps.norm.ppf(1 - self.alpha)
        self.left_side = 1/mean - t_alpha / (np.sqrt(self.n)) / mean
        self.right_side = 1/mean + t_alpha / (np.sqrt(self.n)) / mean

    def print(self):
        print(f"\nExponential trusted interval for lambda={self.sigma}:")
        print(f"({round(self.left_side,3)}, {round(self.right_side,3)})")

class KernelDensityEstimation(ABC):

    @abstractmethod
    def __init__(self, n=500, h=None):
        self.n = n          
        self.h = h   

    @abstractmethod
    def buildKDE(self):
        pass
    
    @abstractmethod
    def plot(self): 
        pass  
    
class GaussKDE(KernelDensityEstimation):

    def __init__(self, n=500, mean=3, dev = np.sqrt(10)):
        self.n = n
        self.mean = mean
        self.dev = dev
        self.h = 1 / (n ** (1/5))

    def buildKDE(self):
        
        np.random.seed(42)
        self.gauss_sample = sps.norm(loc=self.mean, scale=self.dev).rvs(self.n)
        #self.kde_gauss = sps.gaussian_kde(self.gauss_sample, bw_method=self.h)

    def plot(self):
        plt.figure(figsize=(10, 6))
        #x = np.linspace(self.mean - 4*self.dev, self.mean + 4*self.dev, 1000)
        plt.hist(self.gauss_sample, bins=30,density=True, color='green', edgecolor='black', label="histogram")
        sns.kdeplot(self.gauss_sample, bw_method=self.h, fill=True, 
                    label=f"h = {self.h}",linewidth=2, edgecolor="black",color='green')
        
        
        plt.title(f"KDE N")
        plt.legend()
        plt.show()


class UniformKDE(KernelDensityEstimation):

    def __init__(self,n=500, a=-5,b=5):
        self.n = n
        self.a = a
        self.b = b
        self.h = 1 / (n ** (1/5))

    def buildKDE(self):

        np.random.seed(42)
        self.uniform_sample = sps.uniform(loc=self.a, scale=self.b - self.a).rvs(self.n)
        
        #self.kde = KernelDensity(kernel='gaussian', bandwidth=self.h).fit(self.uniform_sample.reshape(-1,1))
        #self.kde_uniform = sps.gaussian_kde(self.uniform_sample, bw_method=self.h)

    ''' def kde_uniform(self,x):
            result = np.zeros_like(x)
            for xi in x:
                result += np.sum(np.where(
                    np.abs((xi - self.uniform_sample) / self.h) <= 1, 0.5, 0))
            return result / (self.n * self.h)
    '''
    
       

    def plot(self):
       
        # plt.figure(figsize=(10, 6))
        # x = np.linspace(self.a, self.b, 1000).reshape(-1,1)
        # x1 = np.linspace(self.a, self.b, 1000)
        # log_dens = self.kde.score_samples(x)
        # dens = np.exp(log_dens)
        
        # plt.plot(x, dens, label=f"h = {self.h}", lw=2)
        # plt.title(f"KDE U")
        # plt.legend()
        # plt.show()
       
        plt.figure(figsize = (10,6))
        #x = np.linspace(self.a, self.b, 1000).reshape(-1,1)
        plt.hist(self.uniform_sample, bins=30,density=True, label="histogram", edgecolor="black", color='green')
        sns.kdeplot(self.uniform_sample, bw_method=self.h, fill=True, label=f"h = {self.h}",
                    linewidth=1, edgecolor="black", color='green')
        
        plt.title(f"KDE U")
        plt.legend()
        plt.show()


gauss = Gauss()
gauss.buildIntervals()
gauss.print()

bernulli = Bernulli()
bernulli.buildIntervals()
bernulli.print()

bernulli50 = Bernulli(50)
bernulli50.buildIntervals()
bernulli50.print()


puasson = Puasson()
puasson.buildIntervals()
puasson.print()

expon = Exponential()
expon.buildIntervals()
expon.print()

kde_gauss = GaussKDE()
kde_gauss.buildKDE()
kde_gauss.plot()

kde_uniform = UniformKDE()
kde_uniform.buildKDE()
kde_uniform.plot()







