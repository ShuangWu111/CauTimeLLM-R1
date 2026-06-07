import json
import os
import argparse
import asyncio
from openai import OpenAI, AsyncOpenAI
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm
from collections import defaultdict
from evaluation_prompts import (
    get_readability_prompt,
    get_logical_consistency_prompt,
    get_comprehensiveness_prompt,
    extract_score
)


# ============== 同步评估函数（保留原有功能）==============

def evaluate_with_Api(reasoning_text, question, label, dimension, client, model, temperature):
    """使用API评估单个维度（同步版本）"""
    try:
        if dimension == 'readability':
            prompt = get_readability_prompt(reasoning_text)
        elif dimension == 'logical_consistency':
            prompt = get_logical_consistency_prompt(reasoning_text, question, label)
        elif dimension == 'comprehensiveness':
            prompt = get_comprehensiveness_prompt(reasoning_text, question)
        else:
            raise ValueError(f"Unknown dimension: {dimension}")

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
        
        response_text = response.choices[0].message.content
        score = extract_score(response_text)
        
        return {
            'score': score,
            'response': response_text
        }
    except Exception as e:
        print(f"Error evaluating {dimension}: {e}")
        return {
            'score': None,
            'response': f"ERROR: {str(e)}"
        }


# ============== 异步评估函数（新增）==============

async def evaluate_with_Api_async(reasoning_text, question, label, dimension, client, model, temperature, semaphore):
    """使用API评估单个维度（异步版本）
    
    Args:
        reasoning_text: 推理文本
        question: 问题
        dimension: 评估维度
        client: AsyncOpenAI 客户端
        model: 模型名称
        temperature: 温度参数
        semaphore: 并发控制信号量
    """
    async with semaphore:  # 使用信号量限制并发
        try:
            if dimension == 'readability':
                prompt = get_readability_prompt(reasoning_text)
            elif dimension == 'logical_consistency':
                prompt = get_logical_consistency_prompt(reasoning_text, question, label)
            elif dimension == 'comprehensiveness':
                prompt = get_comprehensiveness_prompt(reasoning_text, question)
            else:
                raise ValueError(f"Unknown dimension: {dimension}")

            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
            
            response_text = response.choices[0].message.content
            score = extract_score(response_text)
            
            return {
                'score': score,
                'response': response_text
            }
        except Exception as e:
            print(f"Error evaluating {dimension}: {e}")
            return {
                'score': None,
                'response': f"ERROR: {str(e)}"
            }


async def evaluate_single_item_async(item, client, model, temperature, semaphore):
    """异步评估单个样本的所有维度
    
    Args:
        item: 单个样本数据
        client: AsyncOpenAI 客户端
        model: 模型名称
        temperature: 温度参数
        semaphore: 并发控制信号量
        
    Returns:
        包含评估结果的字典
    """
    reasoning_text = item.get('model_output', '')
    question = item.get('question', '')
    label = item.get('label', '')
    dimensions = ['readability', 'logical_consistency', 'comprehensiveness']
    
    # 为每个维度创建任务
    tasks = [
        evaluate_with_Api_async(reasoning_text, question, label, dim, client, model, temperature, semaphore)
        for dim in dimensions
    ]
    
    # 并发执行三个维度的评估
    results = await asyncio.gather(*tasks)
    
    eval_item = {
        'user_segment_id':item['user_segment_id'],
        'window_start_idx':item['window_start_idx'],
        'label': item['label'],
        'predicted_label': item['predicted_label'],
        'evaluations': {}
    }
    
    for dim, result in zip(dimensions, results):
        eval_item['evaluations'][dim] = result
    
    return eval_item


async def process_batch_async(batch, client, model, temperature, semaphore, pbar=None):
    """处理一批样本的异步评估
    
    Args:
        batch: 批次样本列表
        client: AsyncOpenAI 客户端
        model: 模型名称
        temperature: 温度参数
        semaphore: 并发控制信号量
        pbar: 进度条对象
        
    Returns:
        评估结果列表
    """
    # 为批次中的每个样本创建任务
    tasks = [
        evaluate_single_item_async(item, client, model, temperature, semaphore)
        for item in batch
    ]
    
    # 并发执行所有任务
    results = await asyncio.gather(*tasks)
    
    if pbar:
        pbar.update(len(batch))
    
    return results


# ============== 辅助函数 ==============

def load_existing_results(output_file):
    """加载已有的评测结果，用于断点续传"""
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                return json.load(f)
        except:
            return []
    return []


def save_single_result(output_file, evaluation_results):
    """保存单个评测结果到文件"""
    with open(output_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)


# ============== 同步评估主函数（保留原有功能）==============

def run_evaluation(inference_file, output_file, api_key, base_url, model, temperature=0.0):
    """运行评估（边评测边保存，同步版本）

    Args:
        inference_file: 推理结果文件路径
        output_file: 输出文件路径
        api_key: API密钥
        base_url: API基础URL
        model: 使用的模型名称
        temperature: 温度参数（默认0.0）
    """
    print(f"Loading inference results from {inference_file}...")
    with open(inference_file, 'r') as f:
        results = json.load(f)

    # 初始化API客户端
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )

    # 加载已有结果（断点续传）
    evaluation_results = load_existing_results(output_file)
    evaluated_ids = set()
    for item in evaluation_results:
        key = f"{item['user_segment_id']}_{item['window_start_idx']}"
        evaluated_ids.add(key)

    dimensions = ['readability', 'logical_consistency', 'comprehensiveness']

    print(f"Total samples: {len(results)}")
    print(f"Already evaluated: {len(evaluation_results)}")
    print(f"Remaining: {len(results) - len(evaluation_results)}")

    # 过滤掉已评测的样本
    remaining_results = []
    for item in results:
        key = f"{item['user_segment_id']}_{item['window_start_idx']}"
        if key not in evaluated_ids:
            remaining_results.append(item)

    if not remaining_results:
        print("All samples already evaluated!")
    else:
        print(f"\nEvaluating {len(remaining_results)} samples on 3 dimensions...")

        for item in tqdm(remaining_results, desc="Evaluating"):
            reasoning_text = item.get('model_output', '')
            question = item.get('question', '')
            label = item.get('label', '')

            eval_item = {
                'user_segment_id':item['user_segment_id'],
                'window_start_idx':item['window_start_idx'],
                'label': item['label'],
                'predicted_label': item['predicted_label'],
                'evaluations': {}
            }

            for dimension in dimensions:
                eval_result = evaluate_with_Api(reasoning_text, question, label, dimension, client, model, temperature)
                eval_item['evaluations'][dimension] = eval_result

            # 立即保存结果
            evaluation_results.append(eval_item)
            save_single_result(output_file, evaluation_results)

    # 计算统计信息
    stats = calculate_statistics(evaluation_results)

    # 保存统计信息
    stats_file = output_file.replace('.json', '_stats.json')
    print(f"\nSaving statistics to {stats_file}...")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # 打印统计信息
    print_statistics(stats)

    print(f"\n✅ Evaluation completed!")
    print(f"   Detailed results: {output_file}")
    print(f"   Statistics: {stats_file}")


# ============== 异步评估主函数（新增）==============

async def run_evaluation_async(
    inference_file, 
    output_file, 
    api_key, 
    base_url, 
    model, 
    temperature=0.0,
    max_concurrent=10,
    batch_size=5,
    save_interval=10
):
    """运行评估（异步并发版本）

    Args:
        inference_file: 推理结果文件路径
        output_file: 输出文件路径
        api_key: API密钥
        base_url: API基础URL
        model: 使用的模型名称
        temperature: 温度参数（默认0.0）
        max_concurrent: 最大并发请求数（默认10）
        batch_size: 每批处理的样本数（默认5）
        save_interval: 每隔多少个样本保存一次结果（默认10）
    """
    print(f"Loading inference results from {inference_file}...")
    with open(inference_file, 'r') as f:
        results = json.load(f)

    # 初始化异步API客户端
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url
    )

    # 加载已有结果（断点续传）
    evaluation_results = load_existing_results(output_file)
    evaluated_ids = set()
    for item in evaluation_results:
        key = f"{item['user_segment_id']}_{item['window_start_idx']}"
        evaluated_ids.add(key)

    print(f"Total samples: {len(results)}")
    print(f"Already evaluated: {len(evaluation_results)}")
    print(f"Remaining: {len(results) - len(evaluation_results)}")
    print(f"Max concurrent requests: {max_concurrent}")
    print(f"Batch size: {batch_size}")

    # 过滤掉已评测的样本
    remaining_results = []
    for item in results:
        key = f"{item['user_segment_id']}_{item['window_start_idx']}"
        if key not in evaluated_ids:
            remaining_results.append(item)

    if not remaining_results:
        print("All samples already evaluated!")
    else:
        print(f"\nEvaluating {len(remaining_results)} samples on 3 dimensions...")
        
        # 创建信号量控制并发
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # 将剩余样本分批
        batches = [
            remaining_results[i:i + batch_size] 
            for i in range(0, len(remaining_results), batch_size)
        ]
        
        # 创建进度条
        with tqdm(total=len(remaining_results), desc="Evaluating") as pbar:
            for i, batch in enumerate(batches):
                # 处理当前批次
                batch_results = await process_batch_async(
                    batch, client, model, temperature, semaphore, pbar
                )
                
                # 添加到总结果
                evaluation_results.extend(batch_results)
                
                # 定期保存结果
                if (i + 1) % max(1, save_interval // batch_size) == 0:
                    save_single_result(output_file, evaluation_results)
                    print(f"  💾 Saved progress: {len(evaluation_results)}/{len(results)}")
        
        # 最终保存
        save_single_result(output_file, evaluation_results)
        print(f"  💾 Final save: {len(evaluation_results)}/{len(results)}")

    # 关闭异步客户端
    await client.close()

    # 计算统计信息
    stats = calculate_statistics(evaluation_results)

    # 保存统计信息
    stats_file = output_file.replace('.json', '_stats.json')
    print(f"\nSaving statistics to {stats_file}...")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # 打印统计信息
    print_statistics(stats)

    print(f"\n✅ Async evaluation completed!")
    print(f"   Detailed results: {output_file}")
    print(f"   Statistics: {stats_file}")


def run_evaluation_async_wrapper(
    inference_file, 
    output_file, 
    api_key, 
    base_url, 
    model, 
    temperature=0.0,
    max_concurrent=10,
    batch_size=5,
    save_interval=10
):
    """异步评估的包装器函数，方便在同步环境中调用
    
    使用示例:
        run_evaluation_async_wrapper(
            inference_file="path/to/results.json",
            output_file="path/to/output.json",
            api_key="your_api_key",
            base_url="https://api.example.com/v1",
            model="model-name",
            max_concurrent=20,  # 最大并发数
            batch_size=10,      # 每批样本数
            save_interval=50    # 每隔50个样本保存
        )
    """
    asyncio.run(run_evaluation_async(
        inference_file=inference_file,
        output_file=output_file,
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        max_concurrent=max_concurrent,
        batch_size=batch_size,
        save_interval=save_interval
    ))


# ============== 统计函数（保留原有功能）==============

def calculate_statistics(evaluation_results):
    """计算得分分布和平均值（None值使用1.5分，得分范围0-3）"""
    dimensions = ['readability', 'logical_consistency', 'comprehensiveness']
    stats = {}

    for dimension in dimensions:
        scores = []
        distribution = defaultdict(int)
        none_count = 0

        for item in evaluation_results:
            score = item['evaluations'][dimension]['score']
            if score is not None:
                scores.append(score)
                distribution[score] += 1
            else:
                # None值使用1.5分（0-3范围的中间值）
                scores.append(1.5)
                distribution['None (1.5)'] += 1
                none_count += 1


        if scores:
            avg_score = sum(scores) / len(scores)
            stats[dimension] = {
                'average': round(avg_score, 3),
                'distribution': dict(sorted(distribution.items(), key=lambda x: (isinstance(x[0], str), x[0]))),
                'num_samples': len(scores),
                'none_count': none_count
            }
        else:
            stats[dimension] = {
                'average': None,
                'distribution': {},
                'num_samples': 0,
                'none_count': 0
            }

    return stats


def print_statistics(stats):
    """打印统计信息"""
    print("\n" + "="*60)
    print("Evaluation Statistics:")
    print("="*60)

    for dimension, metrics in stats.items():
        print(f"\n{dimension.replace('_', ' ').title()}:")
        print(f"  Average Score: {metrics['average']:.3f}" if metrics['average'] is not None else "  Average Score: N/A")
        print(f"  Samples: {metrics['num_samples']}")
        if metrics.get('none_count', 0) > 0:
            print(f"  None values (counted as 1.5): {metrics['none_count']}")
        print(f"  Distribution:")

        # 先打印数字分数
        max_count = max(metrics['distribution'].values(), default=1)
        for score in range(4):
            count = metrics['distribution'].get(score, 0)
            bar = "█" * int(count / max_count * 30) if count > 0 else ""
            print(f"    {score}: {count:4d} {bar}")

        # 再打印None值
        none_count = metrics['distribution'].get('None (1.5)', 0)
        if none_count > 0:
            bar = "█" * int(none_count / max_count * 30)
            print(f"    None (1.5): {none_count:4d} {bar}")

    print("="*60)


# ============== 主程序 ==============

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run evaluation on AW_FB inference results",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--inference-files", "-i",
        nargs="+",
        default=['/path/to/inference_results_1.json'],
        help="List of inference result files"
    )
    parser.add_argument(
        "--output-files", "-o",
        nargs="+",
        default=['/path/to/eval_output_1.json'],
        help="List of output evaluation files (must match inference files)"
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("key_for_kimik2", "your_api_key_here"),
        help="API key"
    )
    parser.add_argument(
        "--base-url",
        default="https://api.moonshot.cn/v1",
        help="API base URL"
    )
    parser.add_argument(
        "--model",
        default="kimi-k2-0905-preview",
        help="Model name"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature parameter"
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Use synchronous mode (default: async)"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Max concurrent requests for async mode"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for async mode"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=5,
        help="Save interval for async mode"
    )
    args = parser.parse_args()

    if len(args.inference_files) != len(args.output_files):
        print("Error: Number of inference files must match number of output files")
        exit(1)

    for i in range(len(args.inference_files)):
        if not args.sync:
            print(f"\n{'='*60}")
            print(f"Running ASYNC evaluation for file {i+1}/{len(args.inference_files)}")
            print(f"{'='*60}")
            run_evaluation_async_wrapper(
                inference_file=args.inference_files[i],
                output_file=args.output_files[i],
                api_key=args.api_key,
                base_url=args.base_url,
                model=args.model,
                temperature=args.temperature,
                max_concurrent=args.max_concurrent,
                batch_size=args.batch_size,
                save_interval=args.save_interval
            )
        else:
            print(f"\n{'='*60}")
            print(f"Running SYNC evaluation for file {i+1}/{len(args.inference_files)}")
            print(f"{'='*60}")
            run_evaluation(
                inference_file=args.inference_files[i],
                output_file=args.output_files[i],
                api_key=args.api_key,
                base_url=args.base_url,
                model=args.model,
                temperature=args.temperature
            )
