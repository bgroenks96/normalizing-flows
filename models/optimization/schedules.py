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
        self.use_post_warmup_schedule = post_warmup_schedule is not None
        self.post_warmup_schedule = post_warmup_schedule if self.use_post_warmup_schedule else lambda x: x
        self.name = name
        self.initial_learning_rate = tf.convert_to_tensor(initial_learning_rate, name="initial_learning_rate")

    def __call__(self, step):
        dtype = self.initial_learning_rate.dtype
        step = tf.cast(step, dtype)
        p = tf.math.maximum(1.0, step / self.num_warmup_steps)
        return p*self.initial_learning_rate
        

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "num_warmup_steps": self.num_warmup_steps,
            "post_warmup_schedule": self.post_warmup_schedule,
            "name": self.name
        }
