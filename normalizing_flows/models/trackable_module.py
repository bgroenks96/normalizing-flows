import tensorflow as tf

class TrackableModule(tf.Module):
    def __init__(self, additional_objects, **kwargs):
        super().__init__(**kwargs)
        variables = {var.name: var for var in self.trainable_variables}
        self.checkpoint = tf.train.Checkpoint(**variables, **additional_objects)
        
    def checkpoint_num(self):
        return self.checkpoint.save_counter
        
    def save(self, dirpath):
        return self.checkpoint.save(dirpath)
    
    def load(self, dirpath, save_num=0):
        save_num = save_num if save_num > 0 else self.checkpoint_num()
        return self.checkpoint.restore(f'{dirpath}-{save_num}')
    
    def create_checkpoint_manager(self, dirpath, max_to_keep=1):
        return tf.train.CheckpointManager(self.checkpoint, dirpath, max_to_keep)
