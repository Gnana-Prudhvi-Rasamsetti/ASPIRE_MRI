from models import UNet
from .train_expert import train_expert
def train(*args,**kwargs): return train_expert(UNet(),*args,**kwargs)
