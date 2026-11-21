from models import Restormer
from .train_expert import train_expert
def train(*args,**kwargs): return train_expert(Restormer(),*args,**kwargs)
