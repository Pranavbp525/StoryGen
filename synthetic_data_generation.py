import pickle
import json
import codecs
import random
from transformers import pipeline, set_seed


def main():
    with open('prompts.pkl', 'rb') as f:
        prompts = pickle.load(f)

    seed = 42
    set_seed(seed)

    model_name = 'PranavBP525/phi-2-finetuned-1k_stories_100_genre'
    generator = pipeline('text-generation', model=model_name, device=0)

    generated_examples = generate_examples(generator, prompts)

    with open('input_data.json', 'w') as f:
        json.dump(generated_examples, f, indent=2)

    pairs = create_comparison_dataset('preferences.json')
    random.shuffle(pairs)

    with open('preference_data.json', 'w') as f:
        json.dump(pairs, f, indent=2)


def generate_examples(generator, prompt_list, max_length=1000, num_return_sequences=2):
    results = generator(prompt_list, max_length=max_length, num_return_sequences=num_return_sequences, do_sample=True,
                        pad_token_id=50256)  # Ensure EOS token ID is correctly set

    examples = []
    for i, result_group in enumerate(results):
        prompt = prompt_list[i % len(prompt_list)]
        example = {'prompt': prompt}
        if isinstance(result_group, list):
            for j, res in enumerate(result_group):
                answer = res['generated_text'].lstrip().removeprefix(prompt).strip()
                example[f'answer{j + 1}'] = answer
        else:
            answer = result_group['generated_text'].lstrip().removeprefix(prompt).strip()
            example['answer1'] = answer
        examples.append(example)
    return examples


def create_comparison_dataset(path: str):
    with codecs.open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    pairs = []
    for sample in data:
        chosen = None
        rejected = None
        for annotation in sample['annotations']:
            if annotation['result'][0]['value']['selected'] == 'left':
                chosen = sample['data']['prompt'] + '\n' + sample['data']['answer1']
                rejected = sample['data']['prompt'] + '\n' + sample['data']['answer2']
            else:
                chosen = sample['data']['prompt'] + '\n' + sample['data']['answer2']
                rejected = sample['data']['prompt'] + '\n' + sample['data']['answer1']
            pair = {
                'chosen': chosen,
                'rejected': rejected
            }
            pairs.append(pair)
    return pairs


if __name__ == "__main__":
    main()
