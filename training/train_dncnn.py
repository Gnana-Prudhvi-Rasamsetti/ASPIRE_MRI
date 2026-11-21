from models import DnCNN
from .train_expert import train_expert
def train(*args,**kwargs): return train_expert(DnCNN(),*args,**kwargs)
