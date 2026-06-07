import json
import os
import re
import asyncio
import argparse
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as async_tqdm
import time
from pathlib import Path



# def extract_answer(text):
#     """从模型输出中提取boxed答案"""
#     pattern = r'\\boxed\{([^}]+)\}'
#     matches = re.findall(pattern, text)
#     if matches:
#         try:
#             return int(matches[-1])
#         except:
#             return None

#     return None

def extract_answer(text, subtask):
    """从模型输出中提取json答案"""
    if not text:
        return None

    # 1. 【预处理】去除 Markdown 代码块标记 (```json ... ```)
    # 很多模型喜欢用 Markdown 包裹代码，这一步必不可少
    text = re.sub(r'```json\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'```\s*', '', text)
    
    # 2. 【定位】截取真正的 JSON 字符串
    # 模型可能在 JSON 前后说废话（如 "Here is the output: ... "）
    # 我们只关心第一个 '{' 和最后一个 '}' 之间的内容
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    
    if start_idx == -1 or end_idx == -1:
        print(f"Error: No JSON brackets found in text.")
        return None
        
    json_str = text[start_idx : end_idx + 1]

    # 3. 【解析】尝试解析 JSON，包含基础容错
    data = None
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        # 3.1 尝试修复常见错误：中文引号
        try:
            json_str_fixed = json_str.replace("“", '"').replace("”", '"')
            data = json.loads(json_str_fixed)
        except json.JSONDecodeError as e:
            print(f"JSON Parsing failed: {e}")
            # 这里可以添加更高级的修复逻辑（如使用正则表达式修复尾部逗号），视需求而定
            return None

    if not data:
        return None

    # 4. 【查找】灵活构建键名并查找
    # 目标键名通常是 subtask + "_level"，例如 "Fatigue_level"
    # 但为了鲁棒，我们也兼容输入本身就包含 "_level" 的情况
    target_key = subtask if "_level" in subtask else f"{subtask}_level"
    
    # 4.1 优先检查预期路径：data["Answer"][target_key]
    # 根据你的样例，Answer 是最外层的一个 Key
    if "Answer" in data and isinstance(data["Answer"], dict):
        if target_key in data["Answer"]:
            return data["Answer"][target_key]
    print(f"Warning: Key '{target_key}' not found in JSON data.")
    return None


async def inference_single(item, port, semaphore, base_url, model_path, temperature, top_p, max_tokens, max_retries=3):
    """单个样本异步推理"""
    async with semaphore:  # 控制并发数
        client = AsyncOpenAI(
            api_key="EMPTY",
            base_url=base_url.format(port=port),
            timeout=300.0  # 5分钟超时
        )
        
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=model_path,
                    messages=[
                        {"role": "user", "content": item['question']+'/no_think'}
                    ],
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens
                )

                output_text = response.choices[0].message.content
                
                # 检查输出是否只有换行符（去除空白字符后为空）
                if output_text and output_text.strip():
                    # 输出有效，跳出重试循环
                    predicted_label = extract_answer(output_text, item['subtask'])
                    return {
                        'user_id': item['user_id'],
                        'subtask': item['subtask'],
                        'time_frame': item['time_frame'],
                        'question': item['question'],
                        'label': item['label'],
                        'model_output': output_text,
                        'predicted_label': predicted_label
                    }
                else:
                    # 输出只有换行符或为空，需要重试
                    print(f"Warning: Empty or newline-only output on port {port}, attempt {attempt + 1}/{max_retries}")
                    if attempt < max_retries - 1:
                        continue  # 继续下一次重试
                    else:
                        # 已达到最大重试次数
                        print(f"Error: Max retries reached on port {port}, returning empty result")
                        return {
                            'user_id': item['user_id'],
                            'subtask': item['subtask'],
                            'time_frame': item['time_frame'],
                            'question': item['question'],
                            'label': item['label'],
                            'model_output': output_text if output_text else "ERROR: Empty output after max retries",
                            'predicted_label': None
                        }
                        
            except Exception as e:
                print(f"Error processing item on port {port}, attempt {attempt + 1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    continue  # 发生异常时也尝试重试
                else:
                    return {
                        'user_id': item.get('user_id', 'unknown'),
                        'subtask': item.get('subtask', 'unknown'),
                        'time_frame': item.get('time_frame', 'unknown'),
                        'question': item.get('question', ''),
                        'label': item.get('label', None),
                        'model_output': f"ERROR: {str(e)}",
                        'predicted_label': None
                    }


async def run_inference_async(qa_file, output_file, ports=[8001], max_concurrent=32,
                              base_url="http://localhost:{port}/v1",
                              model_path="/path/to/model",
                              temperature=0.6, top_p=0.95, max_tokens=8001):
    """
    异步并发运行推理

    参数:
        qa_file: QA数据文件路径
        output_file: 输出结果文件路径
        ports: vLLM服务端口列表，每个端口对应一个GPU
        max_concurrent: 最大并发请求数
        base_url: vLLM服务URL模板，{port}会被替换为实际端口
        model_path: 模型路径
        temperature: 采样温度
        top_p: nucleus采样参数
        max_tokens: 最大生成token数
    """
    print(f"Loading QA pairs from {qa_file}...")
    with open(qa_file, 'r') as f:
        qa_pairs = json.load(f)

    print(f"Total samples: {len(qa_pairs)}")
    print(f"Using {len(ports)} GPU(s) on ports: {ports}")
    print(f"Max concurrent requests: {max_concurrent}")
    print(f"Model: {model_path}")
    print(f"Temperature: {temperature}, Top-p: {top_p}, Max tokens: {max_tokens}")

    # 创建信号量控制并发数
    semaphore = asyncio.Semaphore(max_concurrent)

    # 创建所有异步任务
    tasks = []
    for idx, item in enumerate(qa_pairs):
        port = ports[idx % len(ports)]  # 轮询分配端口
        task = inference_single(item, port, semaphore, base_url, model_path,
                               temperature, top_p, max_tokens)
        tasks.append(task)

    print(f"Running async inference...")
    start_time = time.time()

    # 使用 tqdm 显示进度
    results = []
    for coro in async_tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Inference"):
        result = await coro
        results.append(result)

    elapsed_time = time.time() - start_time

    print(f"Saving results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    valid_predictions = sum(1 for r in results if r['predicted_label'] is not None)
    print(f"✅ Inference completed!")
    print(f"   Total: {len(results)}")
    print(f"   Valid predictions: {valid_predictions}")
    print(f"   Failed: {len(results) - valid_predictions}")
    print(f"   Time elapsed: {elapsed_time:.2f}s")
    print(f"   Throughput: {len(results)/elapsed_time:.2f} samples/s")

    # 按端口统计
    port_stats = {port: 0 for port in ports}
    for idx in range(len(qa_pairs)):
        port = ports[idx % len(ports)]
        port_stats[port] += 1

    print(f"\nSamples per GPU:")
    for port, count in port_stats.items():
        print(f"   Port {port}: {count} samples")


def run_inference(qa_file, output_file, ports=[8001], max_concurrent=32,
                  base_url="http://localhost:{port}/v1",
                  model_path="/path/to/model",
                  temperature=0.6, top_p=0.95, max_tokens=8001):
    """同步包装函数"""
    asyncio.run(run_inference_async(qa_file, output_file, ports, max_concurrent,
                                    base_url, model_path, temperature, top_p, max_tokens))


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Run inference on QA dataset using vLLM server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 使用默认配置
  python inference.py

  # 指定输入输出文件
  python inference.py --qa-file test.json --output-file results.json

  # 使用多GPU（多端口）
  python inference.py --ports 8000 8001 8002 8003 --max-concurrent 64

  # 完整参数示例
  python inference.py \\
      --qa-file /path/to/qa_data.json \\
      --output-file results.json \\
      --model-path /path/to/model \\
      --ports 8000 8001 \\
      --max-concurrent 32 \\
      --temperature 0.6 \\
      --top-p 0.95 \\
      --max-tokens 5000
        """
    )

    # 数据配置
    parser.add_argument(
        "--qa-file",
        type=str,
        default="/path/to/qa_data.json",
        help="Input QA data file path"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="/path/to/output_results.json",
        help="Output results file path (default: /path/to/output_results.json)"
    )

    # vLLM服务配置
    parser.add_argument(
        "--ports",
        type=int,
        nargs="+",
        default=[8001],
        help="vLLM server port(s), multiple ports for load balancing (default: [8000])"
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:{port}/v1",
        help="vLLM server URL template, {port} will be replaced (default: http://localhost:{port}/v1)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.getenv("MODEL_PATH", "/path/to/model"),
        help="Model path"
    )

    # 推理参数配置
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=24,
        help="Maximum concurrent requests (default: 16)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature (default: 0.6)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling parameter (default: 0.95)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2000,
        help="Maximum tokens to generate (default: 5000)"
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    print("="*60)
    print("Inference Configuration")
    print("="*60)
    print(f"QA file:        {args.qa_file}")
    print(f"Output file:    {args.output_file}")
    print(f"Model path:     {args.model_path}")
    print(f"Ports:          {args.ports}")
    print(f"Base URL:       {args.base_url}")
    print(f"Max concurrent: {args.max_concurrent}")
    print(f"Temperature:    {args.temperature}")
    print(f"Top-p:          {args.top_p}")
    print(f"Max tokens:     {args.max_tokens}")
    print("="*60)
    print()

    run_inference(
        qa_file=args.qa_file,
        output_file=args.output_file,
        ports=args.ports,
        max_concurrent=args.max_concurrent,
        base_url=args.base_url,
        model_path=args.model_path,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens
    )
