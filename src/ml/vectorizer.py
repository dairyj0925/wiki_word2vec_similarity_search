#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 参考资料
#
import abc
import logging
import sys
import os


class pipelineFactory(metaclass=abc.ABCMeta):
    """
    Declare an interface for operations that create abstract product
    objects.
    """

    @abc.abstractmethod
    def Extract_rawdata(self):
        pass

    @abc.abstractmethod
    def Transform_clean(self):
        '''
        从结构化数据训练模型
        :return:
        '''
        pass

    @abc.abstractmethod
    def Load_to_service(self):
        '''
        从结构化数据训练模型
        :return:
        '''
        pass
