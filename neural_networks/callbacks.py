import numpy as np
import ravop.core as R
global lab 
global best,inc

class Callback:
    def __init__(self,callbacks=None,model=None):
        self.callbacks=callbacks
        self.model=model
        for i in self.callbacks:
            i.model=model
        pass

    def on_train_begin(self):
        for i in self.callbacks:
            i.on_train_begin()
        pass

    def on_train_end(self):
        for i in self.callbacks:
            i.on_train_end()
        pass

    def on_epoch_begin(self):
        for i in self.callbacks:
            i.on_epoch_begin()
        pass

    def on_epoch_end(self):
        for i in self.callbacks:
            i.on_epoch_end()
        pass

    def on_batch_begin(self):
        for i in self.callbacks:
            i.on_batch_begin()
        pass

    def on_batch_end(self):
        for i in self.callbacks:
            i.on_batch_end()
    
        


class base_callback:
    def __init__(self,callbacks=None):
        pass

    def on_train_begin(self):
        return None

    def on_train_end(self):
        return None

    def on_epoch_begin(self):
        return None

    def on_epoch_end(self):
        return None

    def on_batch_begin(self):
        return None

    def on_batch_end(self):
        return None






class EarlyStopping(base_callback):
    def __init__(self,monitor='val_loss',patience=10):
        self.monitor=monitor
        self.patience = patience
        self.best_weights = None
        self.model=None
        pass

    def on_train_begin(self):
        # print("early stopping on_train_begin")
        self.monitor_val=self.get_monitor_vals(self.monitor)
        # print(self.monitor_val)
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf
        global lab 
        global best
        lab=[]
        best=[]
        inc=[]
        self._global_train_batch = 0
        self._previous_epoch_iterations = 0
        self._train_accumulated_time = 0
        self.best_weight=None
        self.best_monitor_value=Inf

    def on_train_end(self):
        print("early stopping on_train_end")
        # self.best_monitor_value=losses
        #  self._global_train_batch = np.Inf
        # self._previous_epoch_iterations = get_val_epoch()
        # self._train_accumulated_time = get_val_tat()
        # self.best_weight=None
        # self.best_monitor_value=Inf
        # pass
        # if self.best_monitor_value<self.previous_vals:
        #     pass

    def on_epoch_begin(self):
        # print(self.model.errors)
        self.monitor_val=self.get_monitor_vals(self.monitor)
        print(self.monitor_val)
        print("early stopping on epoch begin")

    def on_epoch_end(self):
        print("early stopping on epoch end")
        self.wait += 1
        # print(self.model.errors)
        # if self.epoch:
        # pass
        # if  (current, self.best):
        #     self.best = current
        #     self.best_epoch = epoch
        #     if self.restore_best_weights:
        #         self.best_weights = self.model.get_weights()
        #     if self.baseline is None or self._is_improvement(current, self.baseline):
        #         self.wait = 0


    def on_batch_begin(self):
        # print(self.model.errors)
        print("on batch begin")

    def on_batch_end(self):
        global lab,best,inc
        loss_after_batch=self.model.loss
        if loss_after_batch<self.best:
            self.best=loss_after_batch
            #get weights  set to the NN
        lab.append(loss_after_batch)
        if self.best not in best:
            best.append(self.best)
        else:
            inc.append(loss_after_batch)
        print("========stats=========\n",loss_after_batch,"\n",best,"========================")
        print("on batch end ")

    def get_monitor_vals(self,monitor):
        vals=self.model.errors
        if monitor == "train_loss":
            return vals['training']
        elif monitor == "val_loss":
            return vals['validation']
        else:
            raise "unknown monitor value"





class ModelCheckpoint(base_callback):
    def __init__(self,monitor='val_loss',patience=10):
        self.monitor=monitor
        self.patience = patience
        self.best_weights = None
        self.model=None
        pass

    def on_train_begin(self):
        # print("early stopping on_train_begin")
        self.monitor_val=self.get_monitor_vals(self.monitor)
        # print(self.monitor_val)
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.Inf
        global lab 
        global best_checkpoint
        lab=[]
        best_checkpoint=[]
        best_checkpoint=[]
        all_iter=[]
        self._global_train_batch = 0
        

    def on_train_end(self):
        print("model checkpoint on_train_end")

    def on_epoch_begin(self):
        # print(self.model.errors)
        self.monitor_val=self.get_monitor_vals(self.monitor)
        print(self.monitor_val)
        print("model checkpoint on epoch begin")

    def on_epoch_end(self):
        print("model checkpoint on epoch end")
        self.wait += 1



    def on_batch_begin(self):
        # print(self.model.errors)
        print("on batch begin")

    def on_batch_end(self):
        global lab,best_checkpoint,all_iter
        loss_after_batch=self.model.loss
        if loss_after_batch<self.best:
            self.best=loss_after_batch
            self.model.save_model(filename="best_model_weights.json")
            #get weights  set to the NN
        lab.append(loss_after_batch)
        if self.best not in best_checkpoint:
            best_checkpoint.append(self.best)
        print("========stats=========\n",loss_after_batch,"\n", best_checkpoint,"========================")
        print("on batch end ")

    def get_monitor_vals(self,monitor):
        vals=self.model.errors
        if monitor == "train_loss":
            return vals['training']
        elif monitor == "val_loss":
            return vals['validation']
        else:
            raise "unknown monitor value"


