import torch
from torch.optim import Optimizer

class APGNAG(Optimizer):

    def __init__(self, params, lr=1, momentum=0.9, dampening=0,
                 weight_decay=0, nesterov=True, gamma=None):
        self.gamma = gamma



        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
            
        super(APGNAG, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(APGNAG, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = 0 #group['dampening']
            nesterov = True #group['nesterov']

            sparse = group['sparse']

            if not sparse:
                for p in group['params']:
                    if p.grad is None: continue

                    d_p = p.grad.data
                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)
                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                            buf.mul_(momentum).add_(d_p)
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(momentum).add_(1 - dampening, d_p)
                        if nesterov:
                            d_p = d_p.add(momentum, buf)
                        else:
                            d_p = buf

                    p.data.add_(-group['lr'], d_p)

            else: # APG
                for p in group['params']:
                    if p.grad is None: continue

                    d_p = p.grad.data

                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        else:
                            buf = param_state['momentum_buffer']

                        buf.mul_(momentum)

                        z = p.data.add(-group['lr'], d_p) #(10)
                        z = self.soft_thresholding(z, group['lr'] * self.gamma)

                        buf.add_(z).add_(-1*p.data) #(11)
                        p.data[:] = 0
                        p.data.add_(z).add_(momentum, buf) #(12)

                    p.data[p.data<0.0] = 0.0

        return loss

    @staticmethod
    def soft_thresholding(input, alpha):
        tmp = torch.abs(input) - alpha
        tmp[tmp<0.0] = 0.0
        return torch.sign(input) * tmp
