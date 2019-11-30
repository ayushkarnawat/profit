import keras.backend as K
from keras.callbacks import TensorBoard

class XTensorBoard(TensorBoard):
    """Extends keras's tensorboard with learning rate (LR)."""

    def __init__(self, **kwargs):
        super(XTensorBoard, self).__init__(**kwargs)

    def on_epoch_end(self, epoch, logs=None):
        # Save learning rate to logs at the end of each epoch. Useful for graphing.
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super(XTensorBoard, self).on_epoch_end(epoch, logs)
        