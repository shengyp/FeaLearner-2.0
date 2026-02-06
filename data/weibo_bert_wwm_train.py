"""
使用 Chinese-BERT-wwm-ext 模型生成微博文本嵌入
⭐ 修改版: 预处理逻辑与其他文件保持一致，删除空帖子和空用户
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
import string
import warnings
import logging
import torch
from transformers import BertTokenizer, BertModel
from multiprocessing import Pool
from functools import partial

# --------------------- 配置参数（基于脚本所在目录解析路径） ---------------------
from pathlib import Path
_SCRIPT_DIR = Path(__file__).resolve().parent          # .../data
_REPO_ROOT = _SCRIPT_DIR.parent                        # 仓库根目录

CSV_FILE = str(_SCRIPT_DIR / "raw_data" / "weibo" / "weibo_data.csv")
OUTPUT_DIR = str(_SCRIPT_DIR)  # 输出到与脚本同级的 data 目录
STOPWORDS_PATH = str(_REPO_ROOT / "data_analy" / "tools_dataset" / "hit_stopwords.txt")
BERT_MODEL_NAME = "hfl/chinese-bert-wwm-ext"
MAX_POST_LENGTH = 128
MAX_USER_POSTS = 120
EMBEDDING_DIM = 768
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_PROCESSES = 4
PROGRESS_INTERVAL = 50
BATCH_SIZE = 32  # 批处理大小，用于加速

# --------------------- 屏蔽警告 ---------------------
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)


# --------------------- 工具函数 ---------------------
def load_stopwords(path):
    """加载停用词表"""
    if not os.path.exists(path):
        print(f"警告：未找到停用词表 {path}，将不使用停用词过滤")
        return set()
    with open(path, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f)


def clean_text(text, stopwords=None):
    """
    ⭐ 统一的文本清洗函数 - 与特征提取和ERNIE版本保持完全一致
    注意：不进行jieba分词，直接让BERT tokenizer处理
    """
    if not text or not isinstance(text, str):
        return ""

    # 1. 移除@用户
    text = re.sub(r'@[\u4e00-\u9fa5\w]+', '', text)
    
    # 2. 移除URL
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # 3. 移除HTML标签
    text = re.sub(r'<.*?>', '', text)
    
    # 4. 移除特殊字符和标点
    punctuation = string.punctuation + '，。！？；："''《》【】()（）[]'
    text = re.sub(r'[{}]+'.format(re.escape(punctuation)), ' ', text)
    
    # 5. 移除数字
    text = re.sub(r'\d+', '', text)
    
    # 6. 合并多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 7. 停用词过滤
    if stopwords and text:
        tokens = text.split()
        filtered_tokens = [t for t in tokens if t not in stopwords]
        text = " ".join(filtered_tokens)
    
    return text


def process_user_data(user_posts_tuple, stopwords):
    """
    处理单个用户的所有帖子（多进程调用）
    ⭐ 修改：过滤空帖子
    """
    user_id, posts, label = user_posts_tuple
    
    cleaned_posts = []
    for post in posts:
        cleaned = clean_text(post, stopwords)
        # ⭐ 只保留非空且非纯空格的帖子
        if cleaned and len(cleaned.strip()) > 0:
            cleaned_posts.append(cleaned)
    
    # ⭐ 如果该用户没有有效帖子，返回None
    if not cleaned_posts:
        return None
    
    return {
        'user': user_id,
        'posts': cleaned_posts[:MAX_USER_POSTS],
        'label': label
    }


def load_data_from_csv(csv_file):
    """
    从CSV文件读取数据并按用户分组，保持CSV中的顺序
    ⭐ 修改：与其他文件的加载逻辑保持一致
    """
    print(f"正在读取CSV文件: {csv_file}...")
    
    try:
        df = pd.read_csv(csv_file, encoding='utf-8')
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        return []
    
    print(f"CSV文件读取完成，共 {len(df)} 条记录")
    
    # 按用户ID分组，保持CSV中用户第一次出现的顺序
    user_data = {}
    user_order = []
    
    for _, row in df.iterrows():
        user_id = str(row['user_id'])
        post_raw = str(row['Post']) if pd.notna(row['Post']) else ""
        label = int(row['label']) if pd.notna(row['label']) else 0
        
        if user_id not in user_data:
            user_order.append(user_id)
            user_data[user_id] = {
                'posts': [],
                'label': label
            }
        
        if post_raw:
            # ⭐ 按换行符分割，与其他文件保持一致
            post_lines = post_raw.split('\n')
            for line in post_lines:
                line = line.strip()
                if line:
                    user_data[user_id]['posts'].append(line)
    
    # 转换为元组列表，用于多进程处理
    user_tuples = [
        (user_id, user_data[user_id]['posts'], user_data[user_id]['label'])
        for user_id in user_order
    ]
    
    print(f"数据分组完成，共 {len(user_tuples)} 个用户（按CSV顺序）")
    return user_tuples


def get_bert_embedding(text, tokenizer, model, device):
    """使用BERT模型将单条文本转换为向量嵌入"""
    if not text or not text.strip():
        return np.zeros(EMBEDDING_DIM)
    
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding='max_length',
        truncation=True,
        max_length=MAX_POST_LENGTH,
        add_special_tokens=True
    )
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        # 使用[CLS] token的嵌入
        embedding = outputs.last_hidden_state[0, 0, :].cpu().numpy()
    
    return embedding


def get_bert_embeddings_batch(texts, tokenizer, model, device, batch_size=BATCH_SIZE):
    """
    批量提取BERT嵌入，提高处理效率
    """
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_POST_LENGTH,
            add_special_tokens=True
        )
        
        # 移动到设备
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 提取嵌入
        with torch.no_grad():
            outputs = model(**inputs)
            # 使用[CLS] token的嵌入（第一个token）
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        all_embeddings.extend(batch_embeddings)
    
    return np.array(all_embeddings)


def generate_user_embeddings(all_users, tokenizer, model, device, use_batch=True):
    """
    生成所有用户的帖子嵌入列表
    ⭐ 添加批处理选项以提高效率
    """
    user_embeddings = []
    
    if use_batch:
        # 批处理模式：收集所有文本，一次性处理
        all_texts = []
        text_to_user_map = []  # 记录每条文本属于哪个用户
        
        for user_idx, user in enumerate(all_users):
            posts = user['posts'][:MAX_USER_POSTS]
            for post_idx, post in enumerate(posts):
                all_texts.append(post)
                text_to_user_map.append((user_idx, post_idx))
        
        total_posts = len(all_texts)
        print(f"\n开始批量生成嵌入向量，共 {total_posts} 条帖子...")
        print(f"使用设备: {device}")
        print(f"批处理大小: {BATCH_SIZE}")
        
        # 批量提取所有嵌入
        all_embeddings = get_bert_embeddings_batch(all_texts, tokenizer, model, device, BATCH_SIZE)
        
        # 按用户组织嵌入
        user_embeddings_dict = {}
        for i, (user_idx, post_idx) in enumerate(text_to_user_map):
            if user_idx not in user_embeddings_dict:
                user_embeddings_dict[user_idx] = []
            user_embeddings_dict[user_idx].append(all_embeddings[i])
            
            # 显示进度
            if (i + 1) % (PROGRESS_INTERVAL * BATCH_SIZE) == 0:
                progress_pct = (i + 1) * 100 / total_posts
                print(f"进度: {i + 1}/{total_posts} ({progress_pct:.1f}%)")
        
        # 按用户顺序构建最终结果
        for user_idx, user in enumerate(all_users):
            user_embeddings.append({
                'user_id': user['user'],
                'embeddings': [np.array(emb) for emb in user_embeddings_dict[user_idx]],
                'label': user['label']
            })
        
        print(f"所有帖子处理完成！共处理 {total_posts} 条帖子")
        
    else:
        # 原始逐条处理模式（保留以防批处理出问题）
        total_posts = sum(len(user['posts']) for user in all_users)
        processed = 0
        
        print(f"\n开始生成嵌入向量，共 {total_posts} 条帖子...")
        print(f"使用设备: {device}")
        
        for user_idx, user in enumerate(all_users):
            posts = user['posts'][:MAX_USER_POSTS]
            embeddings = []
            
            for post in posts:
                embedding = get_bert_embedding(post, tokenizer, model, device)
                embeddings.append(embedding)
                processed += 1
                
                if processed % PROGRESS_INTERVAL == 0:
                    progress_pct = processed * 100 / total_posts
                    print(f"进度: {processed}/{total_posts} ({progress_pct:.1f}%)")
            
            user_embeddings.append({
                'user_id': user['user'],
                'embeddings': embeddings,
                'label': user['label']
            })
        
        print(f"所有帖子处理完成！共处理 {processed} 条帖子")
    
    return user_embeddings


# --------------------- 主函数 ---------------------
def main():
    print("=" * 70)
    print("使用 Chinese-BERT-wwm-ext 生成微博文本嵌入")
    print("⭐ 修改版: 预处理逻辑统一，删除空帖子和空用户")
    print("=" * 70)
    
    # 加载停用词
    stopwords = load_stopwords(STOPWORDS_PATH)
    print(f"停用词表加载完成，共 {len(stopwords)} 个停用词")
    
    # 从CSV文件读取数据
    if not os.path.exists(CSV_FILE):
        print(f"错误：未找到CSV文件 {CSV_FILE}")
        return
    
    user_tuples = load_data_from_csv(CSV_FILE)
    
    if len(user_tuples) == 0:
        print("错误：没有有效的用户数据")
        return
    
    original_user_count = len(user_tuples)
    
    # 多进程处理文本清洗
    print(f"\n使用 {NUM_PROCESSES} 个进程进行文本清洗...")
    with Pool(processes=NUM_PROCESSES) as pool:
        process_func = partial(process_user_data, stopwords=stopwords)
        results = pool.map(process_func, user_tuples)
    
    # ⭐ 过滤无效数据（空用户）
    all_users = [item for item in results if item is not None]
    empty_user_count = original_user_count - len(all_users)
    
    print(f"文本清洗完成:")
    print(f"  - 原始用户数: {original_user_count}")
    print(f"  - 删除空用户数: {empty_user_count}")
    print(f"  - 有效用户数: {len(all_users)}")
    
    if len(all_users) == 0:
        print("错误：所有用户的帖子都为空!")
        return
    
    # 统计每个用户的帖子数
    post_counts = [len(user['posts']) for user in all_users]
    total_posts = sum(post_counts)
    print(f"\n用户帖子统计:")
    print(f"  - 平均帖子数: {np.mean(post_counts):.2f}")
    print(f"  - 最少帖子数: {np.min(post_counts)}")
    print(f"  - 最多帖子数: {np.max(post_counts)}")
    print(f"  - 总帖子数: {total_posts}")
    
    # 加载BERT模型
    print(f"\n正在加载BERT模型: {BERT_MODEL_NAME}...")
    print(f"检测到设备: {DEVICE}")
    if DEVICE == "cuda":
        try:
            print(f"GPU可用: {torch.cuda.get_device_name(0)}")
        except:
            pass
    print("提示：首次运行会自动下载模型，可能需要较长时间...")
    
    try:
        tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
        model = BertModel.from_pretrained(BERT_MODEL_NAME)
        model.to(DEVICE)
        model.eval()
        print(f"BERT模型加载完成！")
        print(f"  - 模型: {BERT_MODEL_NAME}")
        print(f"  - 隐藏层维度: {EMBEDDING_DIM}")
        print(f"  - 词汇表大小: {len(tokenizer)}")
        print(f"  - 设备: {DEVICE}")
    except Exception as e:
        print(f"加载BERT模型失败: {e}")
        print(f"\n请确保已安装 transformers 和 torch：")
        print(f"pip install transformers torch")
        return
    
    # 生成用户帖子嵌入（使用批处理模式）
    user_embeddings = generate_user_embeddings(
        all_users, tokenizer, model, DEVICE, use_batch=True
    )
    
    # ⭐ 验证数据完整性
    print(f"\n数据验证:")
    print(f"  - 用户数量: {len(user_embeddings)}")
    empty_embedding_count = sum(1 for emb in user_embeddings if len(emb['embeddings']) == 0)
    print(f"  - 空嵌入用户数: {empty_embedding_count}")
    
    if empty_embedding_count > 0:
        print(f"  ⚠️ 警告: 发现 {empty_embedding_count} 个用户的嵌入为空!")
    
    # 保存结果
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "user_post_embeddings_bert_wwm.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(user_embeddings, f)
    
    print(f"\n" + "=" * 70)
    print(f"处理完成！")
    print(f"=" * 70)
    print(f"输出文件: {output_path}")
    print(f"包含 {len(user_embeddings)} 个用户的嵌入数据")
    print(f"嵌入维度: {EMBEDDING_DIM}")
    print("=" * 70)


if __name__ == "__main__":
    main()