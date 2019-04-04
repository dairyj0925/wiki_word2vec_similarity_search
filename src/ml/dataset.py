#!/usr/bin/python
# -*- coding: UTF-8 -*-
# 参考资料
# https://segmentfault.com/a/1190000010129248
# https://sourcemaking.com/design_patterns/abstract_factory/python/1
# https://my.oschina.net/datadev/blog/1836529

"""
Provide an interface for creating families of related or dependent
objects without specifying their concrete classes.
"""

import abc
import logging
import os.path
import sys
import os

class ProcessAbsFactory(metaclass=abc.ABCMeta):
    """
    Declare an interface for operations that create abstract product
    objects.
    """

    @abc.abstractmethod
    def raw2structData(self):
        '''
        从原始数据到按行整理后的数据集
        :return:
        '''
        pass


class ProcessFactoryW2V(ProcessAbsFactory):
    """
    Implement the operations to create concrete product objects.
    """

    def raw2structData(self):
        return ConcreteWIKI()



class AbstractDatasets(metaclass=abc.ABCMeta):
    """
    Declare an interface for a type of product object.
    """

    @abc.abstractmethod
    def dataprocess(self):
        pass



class ConcreteWIKI(AbstractDatasets):
    """
    Define a product object to be created by the corresponding concrete
    factory.
    Implement the AbstractProduct interface.
    """
    def __init__(self):
        self.curpath =  os.path.dirname(os.path.realpath(__file__))
        self.data_path = '../datasets/'
        self.zhwiki_bz2 = 'zhwiki-latest-pages-articles.xml.bz2'
        self.zhwiki_raw = 'zhwiki_raw.txt'
        self.zhwiki_raw_t2s = 'zhwiki_t2s.txt'
        self.zhwiki_seg_t2s = 'zhwiki_seg.txt'
        self.embedded_model_t2s = 'checkpoint/w2v/zhwiki_embedding_t2s.model'
        self.embedded_vector_t2s = 'checkpoint/w2v/vector_t2s'

        program = os.path.basename(sys.argv[0])
        self.logger = logging.getLogger(program)
        logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s')
        logging.root.setLevel(level=logging.INFO)
        self.logger.info("running %s" % ' '.join(sys.argv))

    def dataprocess(self):
        inp, outp = sys.argv[1:3]
        from gensim.corpora import WikiCorpus
        import jieba
        import codecs

        import six
        from gensim.models import Word2Vec
        from gensim.models.word2vec import LineSentence
        import multiprocessing

        output = open(os.path.join(self.curpath ,self.data_path, outp), 'w', encoding='utf-8')
        wiki = WikiCorpus(os.path.join(self.curpath ,self.data_path, inp), lemmatize=False, dictionary={})
        i = 0
        for text in wiki.get_texts():
            output.write(' '.join(text) + '\n')
            i += 1
            if i % 10000 == 0:
                self.logger.info('Saved ' + str(i) + ' articles')
        output.close()
        self.logger.info('Finished Saved ' + str(i) + ' articles')

    def zhwiki_segment(self, remove_alpha=True):
        def is_alpha(tok):
            try:
                return tok.encode('ascii').isalpha()
            except UnicodeEncodeError:
                return False
        import jieba
        import codecs
        output = open(os.path.join(self.curpath , self.data_path, self.zhwiki_seg_t2s), 'w', encoding='utf-8')
        self.logger.info('start ...zhwiki_segment')
        i = 0
        with codecs.open(os.path.join(self.curpath ,self.data_path, self.zhwiki_raw_t2s), 'r', encoding='utf-8') as raw_input:
            for line in raw_input:
                line = line.strip()
                i += 1
                print('line ' + str(i))
                text = line.split()
                if True:
                    text = [w for w in text if not is_alpha(w)]
                word_cut_seed = [jieba.cut(t) for t in text]
                tmp = ''
                for sent in word_cut_seed:
                    for tok in sent:
                        tmp += tok + ' '
                tmp = tmp.strip()
                if tmp:
                    output.write(tmp + '\n')
            output.close()


def main():
    # 1 提取wiki数据
    f = ProcessFactoryW2V()
    d = f.raw2structData()
    # d.dataprocess()
    # 2 分词
    d.zhwiki_segment()


if __name__ == "__main__":
    main()