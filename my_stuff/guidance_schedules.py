import torch
import abc

class GuidanceSchedule(abc.ABC):
    def __init__(self, scale) -> None:
        super().__init__()
        self.scale = scale
    
    @abc.abstractmethod
    def __call__(self,t):
        pass

class ConstantSchedule(GuidanceSchedule):
    def __init__(self, scale) -> None:
        super().__init__(scale)
    
    def __call__(self, t, w=None):
        scale = self.scale if w is None else w
        return torch.ones_like(t) * scale

class IntervalSchedule(GuidanceSchedule):
    def __init__(self, scale, left=0., right=1.) -> None:
        super().__init__(scale)
        self.left = left
        self.right = right
    
    def __call__(self, t, w=None):
        scale = self.scale if w is None else w
        return torch.where((t >= self.left) & (t <= self.right), scale, 0.) 

class LinearRampUp(GuidanceSchedule):
    def __init__(self, scale, left=0., right=1.) -> None:
        super().__init__(scale)
        self.left = left
        self.right = right
    
    def __call__(self, t, w=None):
        scale = self.scale if w is None else w
        in_interval = (t >= self.left) & (t <= self.right)
        right_mask  = (t >= self.right)
        left_mask   = (t <= self.left) 

        result = torch.ones_like(t)
        result = torch.where(in_interval, scale, 0.)
        if self.right < 1.:
            result = torch.where(right_mask, scale * (t-1)/(self.right-1), result)
        result = torch.where(left_mask, scale * t/self.left, result)
        
        return result
    
    
class LinearSchedule(GuidanceSchedule):
    def __init__(self, scale) -> None:
        super().__init__(scale)
    
    def __call__(self, t):
        return 2 * self.scale * (1-t) + 1

class InverseLinearSchedule(GuidanceSchedule):
    def __init__(self, scale) -> None:
        super().__init__(scale)
    
    def __call__(self, t):
        return 2 * self.scale * t + 1

class CosineSchedule(GuidanceSchedule):
    def __init__(self, scale) -> None:
        super().__init__(scale)
    
    def __call__(self, t):
        return self.scale * (torch.cos(torch.pi * t) + 1) + 1

class SineSchedule(GuidanceSchedule):
    def __init__(self, scale) -> None:
        super().__init__(scale)
    
    def __call__(self, t):
        return self.scale * ((torch.sin(torch.pi * (t - .5))) + 1) + 1
    
class VShapeSchedule(GuidanceSchedule):
    def __init__(self, scale) -> None:
        super().__init__(scale)
    
    def __call__(self, t):
        # return 2 * self.scale * torch.where(t < .5, 1-t, t) + 1

        return 4 * self.scale * (torch.where(t < .5, 1-t, t) - .5) + 1

class LambdaShapeSchedule(GuidanceSchedule):
    def __init__(self, scale) -> None:
        super().__init__(scale)
    
    def __call__(self, t):
        return 4 * self.scale * torch.where(t > .5, 1-t, t) + 1

def get_guidance_schedule(name, scale, **kwargs):
    if name == 'constant':
        return ConstantSchedule(scale)
    elif name == 'interval':
        return IntervalSchedule(scale=scale, **kwargs)
    elif name == 'linear':
        return LinearSchedule(scale)
    elif name == 'linear-ramp-up':
        return LinearRampUp(scale=scale, **kwargs)
    elif name == 'inv-linear':
        return InverseLinearSchedule(scale)
    elif name == 'cosine':
        return CosineSchedule(scale)
    elif name == 'sine':
        return SineSchedule(scale)
    elif name == 'V':
        return VShapeSchedule(scale)
    elif name == 'inv-V':
        return LambdaShapeSchedule(scale)