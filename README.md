# FeaLearner-2.0
1、UCR模块   

实验结论：对于性能没有提升  

2、moe层数对比  
两层moe vs 三层moe  
使用两层moe，没有UCR 指标：Accuracy: 0.5400 test GP: 0.75 GR: 0.6585365853658537 FS: 0.7012987012987012  
使用三层moe，没有UCR 指标：Accuracy: 0.5400 test GP: 0.8181818181818182 GR: 0.6136363636363636 FS: 0.7012987012987013  

3、reddit_clean.pkl是什么？不像是嵌入？？直接修改文件名出现报错  

4、weibo数据集  
user_post_embeddings.pkl没有提供？自己训练？？  
