#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 参考资料
# https://segmentfault.com/a/1190000010129248
# https://qiita.com/yagays/items/26b1e139b081cf2ad813
# https://github.com/dietmar/gensim_word2vec_loss/blob/master/test_doc2vec_loss.py
# https://www.jianshu.com/p/05fb666a72c4
import abc
import logging
import sys
import os
import numpy as np
from gensim.models.callbacks import CallbackAny2Vec


class w2vFactory(metaclass=abc.ABCMeta):
    """
    Declare an interface for operations that create abstract product
    objects.
    """
    @abc.abstractmethod
    def load_data(self):
        pass

    @abc.abstractmethod
    def train(self):
        '''
        从结构化数据训练模型
        :return:
        '''
        pass

    @abc.abstractmethod
    def save_train_loss_plot(self):
        '''
        保存训练图像
        :return:
        '''
        pass

class callback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''

    def __init__(self):
        self.epoch = 0

    def on_epoch_end(self, model):
        self.epoch += 1

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
        self.logger.info("running {}".format( __class__.__name__))
        import dataset as data
        f = data.ProcessFactoryW2V()
        self.d = f.raw2structData()
        pass

    def load_data(self):
        from gensim.models.word2vec import LineSentence
        self.logger.info("loadData")
        path = os.path.join(self.d.curpath ,self.d.data_path, self.d.zhwiki_seg_t2s)

        self.sentences = LineSentence(path)



    def train(self):
        from gensim.models import Word2Vec
        import multiprocessing
        import pickle
        import json
        self.logger.info("train")


        size = 100
        window = 5
        min_count = 5
        sg = 1
        checkpoint = "./zhwiki_t2s_w2v_{}_{}_{}_{}_checkpoint.model".format(size,window,min_count,sg)
        abs_check_path = os.path.join(self.d.curpath, self.d.ml_path, checkpoint)
        abs_check_info_path = os.path.join(self.d.curpath, self.d.ml_path, './info.pickle')
        epoch_start = 0
        alpha = 0.025
        loss_val_list = []

        if os.path.exists(abs_check_path):
            model = Word2Vec.load(os.path.join(self.d.curpath, self.d.ml_path, checkpoint))
            try:
                with open(abs_check_info_path, 'rb') as handle:
                    info = pickle.load(handle)
                    self.logger.info(json.dumps(info))
                    epoch_start = info['epoch']
                    alpha = info['alpha']
                    loss_val_list = info['loss_val_list']

            except Exception as e:
                pass
            finally:
                self.logger.info(len(loss_val_list))
        else:
            model = Word2Vec(
                         size=size, window=window, min_count=min_count, sg=sg, workers=multiprocessing.cpu_count())

            model.build_vocab(self.sentences)


        alpha_delta = 0.001
        passes = 20


        for epoch in range(epoch_start+1, passes):
            self.logger.info(epoch)
            model.alpha, model.min_alpha = alpha, alpha
            model.train(self.sentences, total_examples=model.corpus_count, epochs=model.iter,compute_loss=True,callbacks=[callback()])

            alpha -= alpha_delta
            loss = model.get_latest_training_loss()
            self.logger.info("loss: {}".format(loss))
            loss_val_list.append(loss)
            model.save(abs_check_path)
            with open(abs_check_info_path, 'wb') as handle:
                pickle.dump({'epoch':epoch,'alpha':alpha,'loss_val_list':loss_val_list}, handle, protocol=pickle.HIGHEST_PROTOCOL)


        x = range(0,passes,1)
        y = loss_val_list
        self.logger.info("x len {}y len {}".format(len(x), len(y)))
        import pandas as pd
        data = pd.DataFrame({"epoch" : x,
                       "loss" : y})
        data.to_csv(os.path.join(self.d.curpath, self.d.data_path, 'loss.csv'))



        model.save(os.path.join(self.d.curpath, self.d.ml_path, self.d.embedded_model_t2s))
        model.wv.save_word2vec_format(os.path.join(self.d.curpath, self.d.ml_path, self.d.embedded_vector_t2s), binary=False)

        self.logger.info("Finished!")
        return model
        pass

    def save_train_loss_plot(self):
        import matplotlib.pyplot as plt

        # f, ax = plt.subplots(1, 1)
        # ax.plot(range(1, EPOCHS + 1), losses)
        # ax.set_xlabel('Epoch')
        # ax.set_ylabel('Loss')
        # f.tight_layout()
        # f.savefig('word2vec_loss.png')
        import pandas as pd

        data = pd.read_csv(os.path.join(self.d.curpath, self.d.data_path, 'loss.csv'))
        axes = data.plot("epoch","loss")
        axes.set_xlabel('Epoch')
        axes.set_ylabel('Loss')
        fig = axes.get_figure()
        fig.savefig(os.path.join(self.d.curpath, self.d.data_path,"loss.pdf"))
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
        self.model.load_data()
        self.model.train()

    def test(self,word):
        from gensim.models import Word2Vec

        semi = ''
        model = Word2Vec.load(os.path.join(self.model.d.curpath, self.model.d.ml_path, self.model.d.embedded_model_t2s))
        try:
            semi = model.wv.most_similar(word, topn=10)
        except KeyError:
            print('The word not in vocabulary!')
        for term in semi:
            print('%s,%s' % (term[0], term[1]))
        pass

    def saveFig(self):
        '''
        存储loss等图像
        :return:
        '''
        self.model.save_train_loss_plot()




def main():
    command, param = sys.argv[1:3]
    # 1 训练
    if command == 'w2v':
        m = WIKIModel()
        p = w2vPipeline(m)
        p.doIt()
    elif command == 'test':
        m = WIKIModel()
        p = w2vPipeline(m)
        p.test(param)
    elif command == 'fig':
        m = WIKIModel()
        p = w2vPipeline(m)
        p.saveFig()
    pass
    sys.exit()

if __name__ == "__main__":
    main()