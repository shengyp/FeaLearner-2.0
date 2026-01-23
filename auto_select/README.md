reddit.py 主训练代码
run_experiments.sh 超参数调试的代码  适配reddit bigdata (帖子序列：五分类与四分类) 

最佳指标记录：
reddit : bert_embeddings.pkl  

bigdata : bigdata_bert_embeddings.pkl  bigdata_bert.py训练得到bigdata的预训练结果  

加载最佳的预训练模型进行复现（测试） ：
python reddit.py \
                --cv_heads 8 \
                --cv_d_model 256 \
                --lr 1e-4 \
                --batch_size 4 \
                --max_len 5 \ 
                --epochs 50 \
                --patience 10 \
                --classnum 4 \
                --use_pretrain True \
                --data_embeddings "../data/bigdata_bert_embeddings.pkl" \
                --data_features "../data_analy/feature_bigdata.csv" 2>&1

最佳结果：Accuracy: 0.5366  
 test GP: 0.7779960707269156 GR: 0.6336 FS: 0.6984126984126984 OE: 0.08130081300813008

sigir 提取 ：bert.py 得到sigir_bert_embeddings.pkl 

Accuracy:  0.9489
 test GP: 0.9766081871345029 GR: 0.9709302325581395 FS: 0.9737609329446064 OE: 0.0

(base) ➜ auto_select python reddit.py \
                --max_len 300 \
                --classnum 2 \
                --use_pretrain True \
                --data_embeddings "../data/sigir_bert_embeddings.pkl" \
                --data_features "../data_analy/feature_sigir.csv" 2>&1

weibo中文数据集： 
特征维度：(7327, 130)
  - POS: 57
  - TF-IDF: 50
  - NRC: 10
  - SUI: 13
train.py
总体指标:
Accuracy: 0.8499
F1-Score: 0.8499
Precision: 0.8499
Recall: 0.8499
================

reddit:  
Heads,Dim,LR,BatchSize,MaxLen,Patience,Accuracy,GP,GR,FS,OE
4,128,1e-4,16,200,10,0.6400,0.7804878048780488,0.7804878048780488,0.7804878048780488,0.1

bigdata:  


消融实验 ablation_study.py
