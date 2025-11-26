import json
import argparse
import re
from pathlib import Path
from typing import List, Dict
import sys

sys.path.append('./LiveCodeBench')

from lcb_runner.benchmarks import load_code_generation_dataset
from lcb_runner.evaluation import codegen_metrics


def load_completions_from_jsonl(jsonl_path: str) -> List[Dict]:
    """Load completions from JSONL file."""
    completions_data = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                completions_data.append(json.loads(line))
    return completions_data


def extract_code_from_markdown(text: str) -> str:
    """Extract Python code from ```python ... ``` blocks."""
    match = re.search(r'```python(.*?)```', text, re.DOTALL)
    return match.group(1) if match else text


# def get_generations_for_n_strategy(completions_data: List[Dict], n: int, strategy: str) -> List[List[str]]:
#     generations_list = []
#     available_powers = [1, 2, 4, 8, 16, 32]
#     relevant_powers = [k for k in available_powers if k <= n]
    
#     for problem_completions in completions_data:
#         problem_generations = []
        
#         for k in relevant_powers:
#             key = f"completion_{strategy}@{k}"
#             if key in problem_completions:
#                 completion = problem_completions[key]
#                 code = extract_code_from_markdown(completion)
#                 problem_generations.append(code)
        
#         generations_list.append(problem_generations[:n])
    
#     return generations_list

def load_completions_map(jsonl_path: str) -> Dict[str, Dict]:
    """
    Load completions and map them by question_id.
    """
    completions_map = {}
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if 'question_id' in data:
                    completions_map[data['question_id']] = data
    return completions_map

def get_aligned_generations(
    dataset, 
    completions_map: Dict[str, Dict], 
    n: int, 
    strategy: str
) -> List[List[str]]:
    
    generations_list = []
    target_key = f"completion_{strategy}@{n}"
    
    for sample in dataset:
        q_id = sample.question_id
        
        if q_id in completions_map:
            comp_data = completions_map[q_id]
            
            if target_key in comp_data:
                val = comp_data[target_key]
                problem_generations = [extract_code_from_markdown(val)]
            
            
        generations_list.append(problem_generations)
    
    return generations_list

def evaluate_all_n(
    completions_path: str,
    release_version: str,
    num_processes: int = 8,
    timeout: int = 10,
    output_dir: str = "results"
):
    print(f"Loading completions from {completions_path}")
    # completions_data = load_completions_from_jsonl(completions_path)
    completions_map = load_completions_map(completions_path)
    
    print(f"Loading LiveCodeBench dataset: {release_version}")
    dataset = load_code_generation_dataset(release_version)
    
    dataset = [sample for sample in dataset if sample.question_id in completions_map]
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Evaluate each strategy
    strategies = ['weighted', 'maj', 'naive']
    n_values = [1, 2, 4, 8, 16, 32]
    
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Evaluating strategy: {strategy}")
        print(f"{'='*60}")
        
        results_per_n = {}
        
        for n in n_values:
            print(f"\nEvaluating n={n} for strategy={strategy}")
            
            # generations_list = get_generations_for_n_strategy(completions_data, n, strategy)
            generations_list = get_aligned_generations(dataset, completions_map, n, strategy)
            if len(generations_list) == 0:
                continue
            
            dataset_samples = [instance.get_evaluation_sample() for instance in dataset]

            metrics, results, metadata = codegen_metrics(
                samples_list=dataset_samples,
                generations_list=generations_list,
                k_list=[n],
                num_process_evaluate=num_processes,
                timeout=timeout,
                debug=True,
            )
            print(f'metrics: {metrics}')
            print("################################")
            print(f'results: {results}')
            import sys
            sys.exit(-1)
            results_per_n[n] = metrics
            if metrics:
                print(f"  Results for n={n}: {metrics}")
        
        # Save results to CSV
        output_file = f"{output_dir}/results_{strategy}.csv"
        print(f"\nSaving results to {output_file}...")
        with open(output_file, "w") as f:
            f.write("n,pass@n\n")
            for n in n_values:
                if n in results_per_n and results_per_n[n]:
                    pass_at_n_key = f"pass@{n}"
                    score = results_per_n[n].get(pass_at_n_key, 0)
                    f.write(f"{n},{score}\n")
                    print(f"  pass@{n}: {score}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate completions for different n values and strategies"
    )
    parser.add_argument(
        "--completions_file",
        type=str,
        default="/fs/nexus-scratch/armins/ThinkPRM/outputs/results/test-time-scaling/livecodebench/Qwen2.5-Coder-7B-Instruct/best_of_n_disc_prm_qwen/best_of_n_completions.jsonl",
        help="Path to JSONL file containing completions",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save results (default: ./results)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="Timeout per test case in seconds (default: 10)",
    )
    
    args = parser.parse_args()
    
    if not Path(args.completions_file).exists():
        print(f"Error: Completions file not found: {args.completions_file}")
        return 1
    
    try:
        evaluate_all_n(
            completions_path=args.completions_file,
            release_version="release_v5",
            num_processes=8,
            timeout=args.timeout,
            output_dir=args.output_dir,
        )
        print("\nEvaluation complete!")
        return 0
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
