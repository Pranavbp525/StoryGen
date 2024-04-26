import warnings

warnings.filterwarnings("ignore")

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from datasets import load_dataset
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer
from transformers import pipeline
from trl import PPOConfig
from peft import LoraConfig, PeftModel
import torch

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

# Assuming you have `bnb_config` set somewhere for BitsAndBytes
# Update this part of your code where `bnb_config` is defined
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,  # Adjusted to use bfloat16 for computation
    bnb_4bit_use_double_quant=False,
)

dataset = load_dataset("json", data_files="ls_input_data.json", split="train")
dataset = dataset.rename_column("prompt", "query")
dataset = dataset.remove_columns(["answer1", "answer2"])

config = PPOConfig(
    model_name="PranavBP525/phi-2-finetuned-1k_stories_100_genre",
    reward_model="PranavBP525/reward_model_outputs",
    learning_rate=1e-3,
    ppo_epochs=4,
    mini_batch_size=8,
    batch_size=8,
    is_peft_model=True
)

# Then, when creating your model, ensure the updated `bnb_config` is used
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    config.model_name,
    device_map="auto",
    peft_config=lora_config,
    quantization_config=bnb_config  # Ensure this argument correctly references your BnB config
)

tokenizer = AutoTokenizer.from_pretrained(config.model_name, padding_side='left')

tokenizer.pad_token = tokenizer.eos_token

reward_model = pipeline("text-classification", model="PranavBP525/reward_model_outputs")


def tokenize(sample):
    sample = tokenizer(sample['query'], padding="max_length", truncation=True, max_length=1024, return_tensors="pt")
    return sample


dataset = dataset.map(tokenize, batched=True)
dataset.set_format(type="torch")


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


from trl import PPOTrainer

ppo_trainer = PPOTrainer(
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

from tqdm import tqdm

generated_texts = []


def generate_story():
    prompt_text = "Generate a 'Science Fiction Thriller' story titled 'The unpredictable AI doom'."
    input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids

    # Generate a sequence
    output_sequences = model.generate(
        input_ids=input_ids,
        max_length=4096,  # Adjust the max length of the generated text

    )

    # Decode and print the generated text
    generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)
    print(f"Prompt: {prompt_text}\nGenerated Story: {generated_text}\n\n\n")
    return generated_text


# Outer loop with description
for epoch in tqdm(range(ppo_trainer.config.ppo_epochs), desc="Epoch Progress", leave=False):
    # Inner loop with leave=False to clear each batch progress after completion
    for batch in tqdm(ppo_trainer.dataloader, desc="Batch Progress", leave=False):
        query_tensors = batch["input_ids"]

        # Get response from SFTModel
        response_tensors = ppo_trainer.generate(query_tensors, **generation_kwargs)
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        # Compute reward score
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = reward_model(texts, return_all_scores=True)
        rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]

        # Step in the PPO training
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)
    generated_texts.append(generate_story())

    ppo_trainer.push_to_hub("phi-2-storygen-rlhf")
