import json

from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from argparse import Namespace, ArgumentParser


def parse_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--test_file", type=str,
                        default=None,
                        help="test data path")
    parser.add_argument("--model_name_or_path", type=str,
                        default=None,
                        help="model name or path")
    parser.add_argument("--strategy", type=str,
                        default=None,
                        help="decoding strategies")
    parser.add_argument("--num_beams", type=int,
                        default=None,
                        help="number of beams")
    parser.add_argument("--top_k", type=int,
                        default=None,
                        help="top_k")
    parser.add_argument("--top_p", type=float,
                        default=None,
                        help="top_p")
    parser.add_argument("--temperature", type=float,
                        default=None,
                        help="temperature")
    parser.add_argument("--output_path", type=str,
                        default=None,
                        help="output_path")
    return parser.parse_args()



def main():
    args = parse_arguments()
    
    params = {}
    do_sample = False
    
    if args.strategy == "greedy":
        params["num_beams"] = 1
    elif args.strategy == "beam":
        if args.num_beams <= 1:
            raise ValueError("num_beams should be greater than 1 for 'beam' strategy.")
        params["num_beams"] = args.num_beams
    elif args.strategy == "top-k":
        if args.top_k < 1:
            raise ValueError("top_k should be greater than or equal to 1 for 'top-k' strategy.")
        params["top_k"] = args.top_k
        do_sample = True
    elif args.strategy == "top-p":
        if not (0 < args.top_p <= 1):
            raise ValueError("top_p should be in the range (0, 1] for 'top-p' strategy.")
        params["top_p"] = args.top_p
        do_sample = True
    elif args.strategy == "temperature":
        if args.temperature <= 0.0:
            raise ValueError("temperature should be greater than 0.0 for 'temperature' strategy.")
        params["temperature"] = args.temperature
        params["num_beams"] = args.num_beams
        do_sample = True
    
    with open(args.test_file, 'r') as file:
        test_data = [json.loads(line) for line in file]
        
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    
    with open(args.output_path, 'w') as f:
        for i in tqdm(range(len(test_data))):
            id = test_data[i]["id"]
            text = test_data[i]["maintext"]

            inputs = tokenizer(text, return_tensors="pt").input_ids
            outputs = model.generate(inputs, max_new_tokens=100, do_sample=do_sample, **params)
            
            predicted_title = tokenizer.decode(outputs[0], skip_special_tokens=True)
            predicted_title = predicted_title.strip()
            
            d = {"id" : id, "title" : predicted_title}
            f.write(json.dumps(d) + '\n')
            

if __name__ == "__main__":
    main()
            
