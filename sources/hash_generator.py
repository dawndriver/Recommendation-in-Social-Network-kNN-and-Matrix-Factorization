import const
import random

# Given number of hash functions, return a list of hash functions
# n = sig_dim
def gen_minhash_params(n):
  params = []
  for i in range(n):
    a = random.randint(1,1000000)
    b = random.randint(1,1000000)
    k = const.minhash_prime
    # print 'func'+str(i)+'='+str(a)+'*x+'+str(b)+'%'+str(k)
    p = [a,b]
    params.append(p)
  return params
