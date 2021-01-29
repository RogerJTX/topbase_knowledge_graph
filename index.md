## Welcome to GitHub Pages

[editor on GitHub](https://github.com/RogerJTX/topbase_knowledge_graph/edit/gh-pages/index.md) 


# My Main Notes Websites

Main Notes Website  
[https://rogerjtx.github.io/](https://rogerjtx.github.io/)

Word2Vector [ELMo, Bert, ALBert]    
[https://rogerjtx.github.io/word2vector.io/](https://rogerjtx.github.io/word2vector.io/) 

Topbase Knowledge Graph Paper Reproduction And Technical Documentation  
[https://rogerjtx.github.io/topbase_knowledge_graph/](https://rogerjtx.github.io/topbase_knowledge_graph/)

Automatic Code Generation  
url:

Comelot Table Image Recognition   
url:

Aminer Expert Know Graph  
url:

Patent Systemt Keyword Extractor    
url:


Image Style Feature Fusion  
url:

Medical Image Recognition [COVID-19]    
url:




# Programming Notes Start

----------------------------------------



# TexSmart技术方案简介

对于分词、词性标注、句法分析等较为成熟的NLP任务，TexSmart实现了多种代表性的方法[1-4]。下面将简要地介绍其特色功能的技术实现。



# 1.细粒度命名实体识别

    现有的命名实体识别（NER）系统大多依赖于一个带有粗粒度实体类型标注的人工标注数据集来作为训练集。而TexSmart中的实体类型多达千种，人工标注一个带有全部类型标注的训练集是非常耗时的。为减少人工标注量，该模块采用了一种混合（hybrid）方法，它是如下三种方法的融合：
    1) 无监督的细粒度实体识别方法，基于两类数据：其一是从腾讯AI Lab所维护的知识图谱TopBase[5]中所导出的实体名到类型的映射表；其二是采用文献[6-7]中的无监督方法从大规模文本数据中所抽取到的词语上下位关系信息。
    2) 有监督的序列标注模型，基于一个经过人工标注的包含十几种粗粒度实体类型的数据集所训练而成。
    3) 腾讯AI Lab在国际大赛夺冠的实体链接方法[8]。
    这三种方法的结果都会有一些错误和缺陷，实验证明三种方法结合起来能够达到更好的效果。



# 2.语义联想
上下文相关的语义联想（context-aware semantic expansion，简称CASE）是腾讯 AI Lab 从工业应用中抽象出的一个新 NLP 任务[9]。该任务的难点在于缺乏有标注的训练数据。该模块采用了两种方法来构建语义联想模型。第一种方法结合词向量技术、分布相似度技术和模板匹配技术来产生一个语义相似度图[10-12]，然后利用相似度图和上下文信息来产生相关的实体集合。另一种方法是基于大规模的无结构化数据构建一个规模相当的伪标注数据集，并训练一个充分考虑上下文的神经网络模型。
特定类型实体的深度语义表达
对于时间和数量两种实体，TexSmart可以推导出它们具体的语义表达。一些NLP工具利用正则表达式或者有监督的序列标注方法来识别时间和数量实体。但是，这些方法很难推导出实体的结构化语义信息。为了克服这个问题，该模块的实现采用了比正则表达式表达能力更强的上下文无关文法（CFG）。基本流程是：先根据特定类型实体的自然语言表达格式来编写CFG的产生式，然后利用Earley算法[13]来把表示这种实体的自然语言文本解析为一棵语法树，最后通过遍历语法树来生成实体的深度语义表达。

# 3.Reference 参考资料

    •	[1]. John Lafferty, Andrew McCallum, and Fernando Pereira. Conditional random fields: Probabilistic models for segmenting and labeling sequence data, ICML 2001.
    •	[2]. Alan Akbik, Duncan Blythe, and Roland Vollgraf . Contextual String Embeddings for Sequence Labeling. COLING 2018.
    •	[3]. Nikita Kitaev and Dan Klein. Constituency Parsing with a Self-Attentive Encoder. ACL 2018.
    •	[4]. Peng Shi and Jimmy Lin. Simple BERT Models for Relation Extraction and Semantic Role Labeling. Arxiv 2019.
    •	[5]. https://www.infoq.cn/article/kYjJqkao020DcHDMJINI 
    •	[6]. Marti A. Hearst. Automatic Acquisition of Hyponyms from Large Text Corpora. ACL 1992. 
    •	[7]. Fan Zhang, Shuming Shi, Jing Liu, Shuqi Sun, Chin-Yew Lin. Nonlinear Evidence Fusion and Propagation for Hyponymy Relation Mining. ACL 2011. 
    •	[8]. https://mp.weixin.qq.com/s/9XXZc4eVzJY7DCpB4Y2MWQ 
    •	[9]. Jialong Han, Aixin Sun, Haisong Zhang, Chenliang Li, and Shuming Shi. CASE: Context-Aware Semantic Expansion. AAAI 2020. 
    •	[10]. Tomas Mikolov, Ilya Sutskever, Kai Chen, Gregory S. Corrado, and Jeffrey Dean. Distributed Representations of Words and Phrases and their Compositionality. NIPS 2013. 
    •	[11]. Yan Song, Shuming Shi, Jing Li, and Haisong Zhang. Directional Skip-Gram: Explicitly Distinguishing Left and Right Context for Word Embeddings. NAACL 2018. 
    •	[12]. Shuming Shi, Huibin Zhang, Xiaojie Yuan, and Ji-Rong Wen. Corpus-based Semantic Class Mining: Distributional vs. Pattern-Based Approaches. COLING 2010. 
    •	[13]. Jay Earley. An Efficient Context-Free Parsing Algorithm. Communications of the ACM, 13(2), 94-102, 1970. 













### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/RogerJTX/topbase_knowledge_graph/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
