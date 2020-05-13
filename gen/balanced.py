import numpy as np
import random
from collections import defaultdict

class BalancedDist(object):
    def __init__(self,y):
        self.by_cat=sort_by_cat(y)
        self.n_cats=len(self.by_cat)

    def in_cat(self,cat_i):
        return np.random.choice(self.by_cat[cat_i])

    def out_cat(self,cat_i):
        j=random.randint(0,self.n_cats-2)
        if(j>=cat_i):
            j+=1
        return self.in_cat(j)

    def sample(self,n,cat):
        return [ self.in_cat(cat) for i in range(n)]

def sort_by_cat(y):
    by_cat=defaultdict(lambda :[])
    for i,y_i in enumerate(y):
        by_cat[y_i].append(i)
    return by_cat