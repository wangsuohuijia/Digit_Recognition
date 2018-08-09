import keras
from config import num_batch_record

# 定义callback类，用于记录每一步epoch之后的test loss和test acc
class TestEpochCallback(keras.callbacks.Callback):
    def __init__(self, test_x, test_y):
        self.train_losses = []
        self.train_acc = []

        self.val_losses = []
        self.val_acc = []

        self.test_losses = []
        self.test_acc = []

        self.test_x = test_x
        self.test_y = test_y

    def on_epoch_end(self, epoch, logs={}):  # batch 为index, logs为当前batch的日志acc, loss...
        self.train_losses.append(logs.get('loss'))
        self.train_acc.append(logs.get('acc'))

        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))

        loss, acc = self.model.evaluate(self.test_x, self.test_y, verbose=0)
        self.test_losses.append(loss)
        self.test_acc.append(acc)

class TestBatchCallback(keras.callbacks.Callback):
    def __init__(self, val_x, val_y, test_x, test_y):
        self.train_losses = []
        self.train_acc = []

        self.val_losses = []
        self.val_acc = []

        self.test_losses = []
        self.test_acc = []

        self.val_x = val_x
        self.val_y = val_y
        self.test_x = test_x
        self.test_y = test_y

        self.batch_steps = 0

    def on_batch_end(self, batch, logs=None): # batch 为index, logs为当前batch的日志acc, loss...
        if self.batch_steps % num_batch_record == 0:
            self.train_losses.append(logs.get('loss'))
            self.train_acc.append(logs.get('acc'))

            val_loss, val_acc = self.model.evaluate(self.val_x, self.val_y, verbose=0)
            self.val_losses.append(val_loss)
            self.val_acc.append(val_acc)

            loss, acc = self.model.evaluate(self.test_x, self.test_y, verbose=0)
            self.test_losses.append(loss)
            self.test_acc.append(acc)

            print('\nbatch-%d record done!' % (self.batch_steps))

        self.batch_steps += 1

    def on_train_end(self, logs=None):
        train_loss = self.model.model.history.history['loss'][-1]
        train_acc = self.model.model.history.history['acc'][-1]
        self.train_losses.append(train_loss)
        self.train_acc.append(train_acc)

        val_loss, val_acc = self.model.evaluate(self.val_x, self.val_y, verbose=0)
        self.val_losses.append(val_loss)
        self.val_acc.append(val_acc)

        loss, acc = self.model.evaluate(self.test_x, self.test_y, verbose=0)
        self.test_losses.append(loss)
        self.test_acc.append(acc)

        print('\nbatch-%d record done!' % (self.batch_steps))
        self.batch_steps = 0
