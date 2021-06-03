from jax.experimental import optimizers


def Optimizer(
        optimizer: str = 'adam',
        exp_decay_lr: bool = True,
        step_size: float = 0.05,
        decay_steps: int = 50,
        decay_rate: float = 0.5,
):
    
    assert optimizer in ['adam',
                         'sgd',
                         'adagrad',
                         'adamax',
                         'momentum',
                         'nesterov',
                         'rmsprop',
                         'rmsprop_momentum'], 'non-standard optimizer choice'
    
    if exp_decay_lr:
        step_schedule = optimizers.exponential_decay(step_size=step_size,
                                                     decay_steps=decay_steps,
                                                     decay_rate=decay_rate)
    else:
        step_schedule = step_size

    # Use optimizers to set optimizer initialization and update functions
    if optimizer == 'adam':
        return optimizers.adam(step_schedule, b1=0.9, b2=0.999, eps=1e-08)

    elif optimizer == 'sgd':
        return optimizers.sgd(step_schedule)
        
    elif optimizer == 'adagrad':
        return optimizers.adagrad(step_schedule)
        
    elif optimizer == 'adamax':
        return optimizers.adamax(step_schedule, b1=0.9, b2=0.999, eps=1e-08)
        
    elif optimizer == 'momentum':
        return optimizers.momentum(step_schedule, mass=1e-02)
        
    elif optimizer == 'nesterov':
        return optimizers.nesterov(step_schedule, mass=1e-02)
        
    elif optimizer == 'rmsprop':
        return optimizers.rmsprop(step_schedule, gamma=0.9, eps=1e-08)
        
    elif optimizer == 'rmsprop_momentum':
        return optimizers.rmsprop_momentum(step_schedule, gamma=0.9, eps=1e-08, momentum=0.9)
