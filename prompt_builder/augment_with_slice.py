import os
import json
import argparse
import subprocess
import multiprocessing as mp
from tqdm import tqdm
from functools import partial
from utils import str2bool

CHUNK_SIZE = 10
QUERY_LENGTH = 10  # last N lines from prompt will be query

# Adjust this path to where your raw repositories are stored
repository_root = "/home/perer876/workspace/amazon-science/cceval/data/python_rawdata"

input_files = {
    "python": "../data/python/line_completion.jsonl",
    "java": "../data/java/line_completion.jsonl",
    "typescript": "../data/typescript/line_completion.jsonl",
    "csharp": "../data/csharp/line_completion.jsonl"
}

file_ext = {"python": "py", "java": "java", "typescript": "ts", "csharp": "cs"}


def run_retriever_cli(query: str, repo_name: str, top_k: int, max_chunk_size: int):
    """
    Invokes the external code-retriever CLI tool.
    """
    cmd = [
        "slice", "semanthic-search",
        "--max-top-k", str(top_k),
        "--max-chunk-size", str(max_chunk_size),
        "--format", "json",
        query,
        repo_name
    ]

    try:
        # Run subprocess and capture stdout
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        return json.loads(result.stdout)
    except subprocess.CalledProcessError as e:
        # print(f"CLI Error for {repo_name}: {e.stderr}")
        return []
    except json.JSONDecodeError:
        # print(f"JSON Decode Error for {repo_name}. Output: {result.stdout}")
        return []


def format_cross_file_context(retrieved_items, language):
    """
    Formats the retrieved items into the prompt string expected by the benchmark.
    """
    line_start_sym = "#" if language == "python" else "//"

    cfc_list = []
    cfc_text = f"{line_start_sym} Here are some relevant code fragments from other files of the repo:\n\n"

    for item in retrieved_items:
        chunk_content = item.get("code_snippet", "")
        filename = item.get("file_path", "")
        score = item.get("score", 0.0)

        cfc_list.append({
            "retrieved_chunk": chunk_content,
            "filename": filename,
            "score": score
        })

        cfc_text += f"{line_start_sym} the below code fragment can be found in:\n{line_start_sym} {filename}\n"
        cfc_text += "\n".join([f"{line_start_sym} {cl}" for cl in chunk_content.strip('\n').splitlines()]) + "\n\n"

    return cfc_list, cfc_text


def get_cfc(example, args):
    """
    Worker function to process a single example.
    """
    repo_name = example["metadata"]["repository"]
    repo_dir = os.path.join(repository_root, repo_name)

    status = None
    if not os.path.isdir(repo_dir):
        example["crossfile_context"] = {}
        status = "project_not_found"
        return example, status

    prompt = example["prompt"]
    groundtruth = example["groundtruth"]

    if args.query_type == "groundtruth":
        # oracle experiment
        prompt_lines = [pl for pl in prompt.split("\n") if pl.strip()]
        groundtruth_lines = [gt for gt in groundtruth.split("\n") if gt.strip()]
        code_lines = prompt_lines + groundtruth_lines
        query = "\n".join(code_lines[-QUERY_LENGTH:])
    elif args.query_type == "last_n_lines":
        prompt_lines = [pl for pl in prompt.split("\n") if pl.strip()]
        query = "\n".join(prompt_lines[-QUERY_LENGTH:])
    else:
        raise NotImplementedError

    retrieved_results = run_retriever_cli(
        query=query,
        repo_name=repo_name,
        top_k=args.maximum_cross_file_chunk,
        max_chunk_size=CHUNK_SIZE
    )

    if not retrieved_results:
        example["crossfile_context"] = {}
        status = "no_crossfile_context"
    else:
        cfc_list, cfc_text = format_cross_file_context(retrieved_results, args.language)

        example["crossfile_context"] = {}
        example["crossfile_context"]["text"] = cfc_text
        example["crossfile_context"]["list"] = cfc_list

    return example, status


def attach_data(args, srcfile):
    empty_cfc = 0
    error_freq = {
        "project_not_found": 0,
        "no_crossfile_context": 0
    }
    output_examples = []

    examples = []
    with open(srcfile) as f:
        for line in f:
            examples.append(json.loads(line))

    pool = mp.Pool(args.num_processes)
    worker = partial(get_cfc, args=args)

    with tqdm(total=len(examples)) as pbar:
        for (d, stat) in pool.imap_unordered(worker, examples):
            if stat in error_freq:
                error_freq[stat] += 1

            if not d.get("crossfile_context") or len(d["crossfile_context"]) == 0:
                empty_cfc += 1
                if not args.skip_if_no_cfc:
                    output_examples.append(d)
            else:
                output_examples.append(d)
            pbar.update()

    print("Total examples with empty CFC: ", empty_cfc)
    print(error_freq)
    return output_examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--query_type",
        type=str,
        default="last_n_lines",
        choices=["last_n_lines", "groundtruth"],
        help="how to form query from prompt"
    )
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        choices=["java", "python", "typescript", "csharp"],
        help="language name"
    )
    parser.add_argument(
        "--output_file_suffix",
        type=str,
        default=None,
        help="add a suffix string to the output file"
    )
    parser.add_argument(
        "--skip_if_no_cfc",
        type=str2bool,
        default=True,
        help="skip adding examples if there is no crossfile context"
    )
    parser.add_argument(
        "--maximum_cross_file_chunk",
        type=int,
        default=10,
        help="max chunks to return (mapped to --max-top-k)"
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=10,
        help="Number of parallel subprocesses to spawn"
    )

    args = parser.parse_args()

    args.output_file_suffix = "" if args.output_file_suffix is None else f"_{args.output_file_suffix}"

    input_file = input_files[args.language]
    output_path = os.path.dirname(input_file)
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    output_filename = f"{base_name}{args.output_file_suffix}.jsonl"
    output_file = os.path.join(output_path, output_filename)

    print(f"Reading from: {input_file}")
    print(f"Writing to: {output_file}")

    output_examples = attach_data(args, input_file)

    with open(output_file, "w") as fw:
        for ex in output_examples:
            fw.write(json.dumps(ex))
            fw.write("\n")