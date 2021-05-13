import numpy as np
import torch

class Metric:
    def __init__(self):
        pass

    def __call__(self, outputs, target, loss):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError


class AccumulatedAccuracyMetric(Metric):
    """
    Works with classification model
    """

    def __init__(self):
        self.correct = 0
        self.total = 0

    def __call__(self, outputs, target, loss):
        pred = outputs[0].data.max(1, keepdim=True)[1]
        self.correct += pred.eq(target[0].data.view_as(pred)).cpu().sum()
        self.total += target[0].size(0)
        return self.value()

    def reset(self):
        self.correct = 0
        self.total = 0

    def value(self):
        return 100 * float(self.correct) / self.total

    def name(self):
        return 'Accuracy'


class AverageNonzeroTripletsMetric(Metric):
    '''
    Counts average number of nonzero triplets found in minibatches
    '''

    def __init__(self):
        self.values = []

    def __call__(self, outputs, target, loss):
        self.values.append(loss[1])
        return self.value()

    def reset(self):
        self.values = []

    def value(self):
        return np.mean(self.values)

    def name(self):
        return 'Average nonzero triplets'

class LossMetric(Metric):
    '''
    Counts average number of nonzero triplets found in minibatches
    '''

    def __init__(self):
        self.values = []

    def __call__(self, outputs, target, loss):
        self.values.append(loss)
        return self.value()

    def reset(self):
        self.values = []

    def value(self):
        return np.mean(self.values)

    def name(self):
        return 'Loss'

class AccuracyTripletsMetric(Metric):
    '''
    Counts average number of nonzero triplets found in minibatches
    '''

    def __init__(self):
        self.values = []
        self.tp = 0
        self.fp = 0
        self.total_p = 0
        self.total_n = 0

    def __call__(self, outputs, target, loss):
        """print(len(outputs))
        print(outputs[0].shape)
        print(outputs[0][0,:])"""
        with torch.no_grad():
          TA = np.linalg.norm(outputs[0] - outputs[1], axis=1)
          TAc = TA[TA < 0.8]
          self.tp = len(TAc)
          self.total_p = len(TA)
          #print("TA:", self.tp)
          FA = np.linalg.norm(outputs[0] - outputs[2], axis=1)
          FAc = FA[FA < 0.8]
          self.fp = len(FAc)
          self.total_n = len(FA)
          self.fn = self.total_p - self.tp
          self.tn = self.total_n - self.fp
          #print("FA:", self.fp)

        accuracy = (self.tp + self.tn) / (self.total_p + self.total_n)
        #print("Acc: ", accuracy)
        self.values.append(accuracy)
        return self.value()

    def reset(self):
        self.values = []
        self.fp = 0
        self.tp = 0
        self.total_p = 0
        self.total_n = 0

    def value(self):
        return np.mean(self.values)

    def name(self):
        return 'Tripplet accuracy'


class RecallTripletsMetric(Metric):
    '''
    Counts average number of nonzero triplets found in minibatches
    '''

    def __init__(self):
        self.values = []
        self.tp = 0
        self.fp = 0
        self.total_p = 0
        self.total_n = 0

    def __call__(self, outputs, target, loss):
        """print(len(outputs))
        print(outputs[0].shape)
        print(outputs[0][0,:])"""
        with torch.no_grad():
          TA = np.linalg.norm(outputs[0] - outputs[1], axis=1)
          TAc = TA[TA < 0.8]
          self.tp = len(TAc)
          self.total_p = len(TA)
          #print("TA:", self.tp)
          FA = np.linalg.norm(outputs[0] - outputs[2], axis=1)
          FAc = FA[FA < 0.8]
          self.fp = len(FAc)
          self.total_n = len(FA)
          self.fn = self.total_p - self.tp
          self.tn = self.total_n - self.fp
          #print("FA:", self.fp)

        recall = self.tp / (self.tp + self.fn)
        #print("Acc: ", accuracy)
        self.values.append(recall)
        return self.value()

    def reset(self):
        self.values = []
        self.fp = 0
        self.tp = 0
        self.total_p = 0
        self.total_n = 0

    def value(self):
        return np.mean(self.values)

    def name(self):
        return 'Tripplet recall'



class FPRTripletsMetric(Metric):
    '''
    Counts average number of nonzero triplets found in minibatches
    '''

    def __init__(self):
        self.values = []
        self.tp = 0
        self.fp = 0
        self.total_p = 0
        self.total_n = 0

    def __call__(self, outputs, target, loss):
        """print(len(outputs))
        print(outputs[0].shape)
        print(outputs[0][0,:])"""
        with torch.no_grad():
          TA = np.linalg.norm(outputs[0] - outputs[1], axis=1)
          TAc = TA[TA < 0.8]
          self.tp = len(TAc)
          self.total_p = len(TA)
          #print("TA:", self.tp)
          FA = np.linalg.norm(outputs[0] - outputs[2], axis=1)
          FAc = FA[FA < 0.8]
          self.fp = len(FAc)
          self.total_n = len(FA)
          self.fn = self.total_p - self.tp
          self.tn = self.total_n - self.fp
          #print("FA:", self.fp)

        fpr = self.fp / (self.fp + self.tn)
        #print("Acc: ", accuracy)
        self.values.append(fpr)
        return self.value()

    def reset(self):
        self.values = []
        self.fp = 0
        self.tp = 0
        self.total_p = 0
        self.total_n = 0

    def value(self):
        return np.mean(self.values)

    def name(self):
        return 'Tripplet fpr'

