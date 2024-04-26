import warnings
import torch
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
from transformers import pipeline
from trl import AutoModelForCausalLMWithValueHead
from trl import PPOConfig
from trl import PPOTrainer


def generate_story(tokenizer, model, story_prompt):
    prompt_text = story_prompt
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids

    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=2048,

    )

    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    print(f"Prompt: {prompt_text}\nGenerated Story: {generated_text}\n\n\n")
    return generated_text


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def ppo_trainer(prompt_dataset, sft_model, reward_model_name, story_prompt):
    lora_config = LoraConfig(
        r=4,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            'q_proj',
            'v_proj',
        ]
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    dataset = load_dataset("json", data_files=prompt_dataset, split="train")
    dataset = dataset.rename_column("prompt", "query")
    dataset = dataset.remove_columns(["answer1", "answer2"])

    config = PPOConfig(
        model_name=sft_model,
        reward_model=reward_model_name,
        learning_rate=1e-3,
        ppo_epochs=4,
        mini_batch_size=8,
        batch_size=8,
        is_peft_model=True
    )

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_name,
        device_map="auto",
        peft_config=lora_config,
        quantization_config=bnb_config  # Ensure this argument correctly references your BnB config
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, padding_side='left')

    tokenizer.pad_token = tokenizer.eos_token

    reward_model = pipeline("text-classification", model=reward_model_name)

    def tokenize(sample):
        sample = tokenizer(sample['query'], padding="max_length", truncation=True, max_length=1024, return_tensors="pt")
        return sample

    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(type="torch")

    trainer = PPOTrainer(
        model=model,
        config=config,
        dataset=dataset,
        tokenizer=tokenizer,
        data_collator=collator
    )

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 1024
    }

    generated_texts = []

    for epoch in tqdm(range(trainer.config.ppo_epochs), desc="Epoch Progress"):
        for batch in tqdm(trainer.dataloader, desc="Batch Progress"):
            query_tensors = batch["input_ids"]

            response_tensors = trainer.generate(query_tensors, **generation_kwargs)
            batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

            texts = [q + r for q, r in zip(batch["query"], batch["response"])]
            pipe_outputs = reward_model(texts, return_all_scores=True)
            rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]

            stats = trainer.step(query_tensors, response_tensors, rewards)
            trainer.log_stats(stats, batch, rewards)
        generated_texts.append(generate_story(tokenizer, model, story_prompt))


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    prompt_dataset = "input_data.json"
    sft_model = "PranavBP525/phi-2-finetuned-1k_stories_100_genre"
    reward_model_name = "PranavBP525/reward_model_outputs"
    story_prompt = "Generate a 'Science Fiction Thriller' story titled 'The unpredictable AI doom'."

    ppo_trainer(prompt_dataset, sft_model, reward_model_name, story_prompt)
