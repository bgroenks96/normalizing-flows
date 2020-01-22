import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

class LinearWarmupSchedule(LearningRateSchedule):
    """A LearningRateSchedule that uses an exponential decay schedule."""

    def __init__(self,
                 initial_learning_rate,
                 num_warmup_steps,
                 post_warmup_schedule: LearningRateSchedule=None,
                 name=None):
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.num_warmup_steps = num_warmup_steps
        self.post_warmup_schedule = post_warmup_schedule
        self.name = name
        self.initial_learning_rate = tf.convert_to_tensor(initial_learning_rate, name="initial_learning_rate")

    def __call__(self, step):
        dtype = self.initial_learning_rate.dtype
        step = tf.cast(step, dtype)
        if step > self.num_warmup_steps and self.post_warmup_schedule is not None:
            return self.post_warmup_schedule(step - self.num_warmup_steps)
        elif step > self.num_warmup_steps:
            return self.initial_learning_rate
        else:
            p = step / self.num_warmup_steps
            return p*self.initial_learning_rate
        

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "num_warmup_steps": self.num_warmup_steps,
            "post_warmup_schedule": self.post_warmup_schedule,
            "name": self.name
        }
