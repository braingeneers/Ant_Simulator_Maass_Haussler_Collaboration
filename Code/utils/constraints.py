import tensorflow as tf

class MinMax(tf.keras.constraints.Constraint):
    """Constrains weight tensors to be in a given interval."""

    def __init__(self, min, max):
        self.min = min
        self.max = max

    def __call__(self, w):
        return tf.clip_by_value(w, self.min, self.max)

    def get_config(self):
        return {'min': self.min, 'max': self.max}
