import tensorflow as tf

class TrackableModule(tf.Module):
    def __init__(self, additional_objects, **kwargs):
        super().__init__(**kwargs)
        self.additional_objects = additional_objects
        self.checkpoint = None
        
    def _init_checkpoint(self):
        variables = dict()
        for var in self.variables:
            k,v = var.name, var
            assert k not in variables, f'found duplicate variable {k}'
            variables[k] = v
        for k, v in self.additional_objects.items():
            variables[k] = v
        self.checkpoint = tf.train.Checkpoint(**variables)
        
    def checkpoint_num(self):
        assert self.checkpoint is not None, 'checkpoint not initialized'
        return self.checkpoint.save_counter
        
    def save(self, dirpath):
        assert self.checkpoint is not None, 'checkpoint not initialized'
        return self.checkpoint.save(dirpath)
    
    def load(self, path, save_num=0):
        assert self.checkpoint is not None, 'checkpoint not initialized'
        save_num = save_num if save_num > 0 else self.checkpoint_num()
        return self.checkpoint.restore(f'{path}-{save_num}')
    
    def create_checkpoint_manager(self, dirpath, max_to_keep=1):
        assert self.checkpoint is not None, 'checkpoint not initialized'
        return tf.train.CheckpointManager(self.checkpoint, dirpath, max_to_keep)