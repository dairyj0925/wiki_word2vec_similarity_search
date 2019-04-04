#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 参考资料
# https://segmentfault.com/a/1190000010129248
import abc
import logging
import sys
import os

class w2vFactory(metaclass=abc.ABCMeta):
    """
    Declare an interface for operations that create abstract product
    objects.
    """

    @abc.abstractmethod
    def loadData(self):
        pass

    @abc.abstractmethod
    def train(self):
        '''
        从结构化数据训练模型
        :return:
        '''
        pass

class WIKIModel(w2vFactory):
    """
    Define a product object to be created by the corresponding concrete
    factory.
    Implement the AbstractProduct interface.
    """
    def __init__(self):
        self.logger = logging.getLogger( __class__.__name__)
        logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
        logging.root.setLevel(level=logging.INFO)
        self.logger.info("running %s" % ' '.join(sys.argv))
        pass

    def loadData(self):
        self.logger.info("loadData")
        pass

    def train(self):
        self.logger.info("train")
        pass


class PipelineDirector(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def doIt(self):
        '''
        导演类
        :return:
        '''
        pass

class w2vPipeline(PipelineDirector):

    def __init__(self,model):
        self.model = model

    def doIt(self):
        '''
        导演类
        :return:
        '''
        self.model.loadData()
        self.model.train()




def main():
    # 1 训练
    m = WIKIModel()
    p = w2vPipeline(m)
    p.doIt()
    pass


if __name__ == "__main__":
    main()