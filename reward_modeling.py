from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from trl import RewardTrainer, RewardConfig, get_peft_config
from datasets import load_dataset
from peft import LoraConfig, TaskType, PeftModel, prepare_model_for_kbit_training, get_peft_model
import torch

from datasets import load_dataset

dataset = load_dataset('json', data_files="preference_data.json")

model_name_or_path = "PranavBP525/phi-2-finetuned-1k_stories_100_genre"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForSequenceClassification.from_pretrained(
    model_name_or_path,
    num_labels=1,
    quantization_config=bnb_config,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=4,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_CLS",
    target_modules=[
        'q_proj',
        'v_proj',
    ]
)
model = get_peft_model(model, peft_config)


def preprocess_function(examples):
    # Tokenize both chosen and rejected examples
    tokenized_chosen = tokenizer(examples["chosen"], padding="max_length", truncation=True, max_length=1024)
    tokenized_rejected = tokenizer(examples["rejected"], padding="max_length", truncation=True, max_length=1024)

    # Return formatted inputs for training
    return {
        "input_ids_chosen": tokenized_chosen["input_ids"],
        "attention_mask_chosen": tokenized_chosen["attention_mask"],
        "input_ids_rejected": tokenized_rejected["input_ids"],
        "attention_mask_rejected": tokenized_rejected["attention_mask"],
    }


dataset = dataset.map(preprocess_function, batched=True)

model.config.pad_token_id = tokenizer.pad_token_id
reward_config = RewardConfig(
    output_dir="./reward_model_outputs",
    per_device_train_batch_size=32,
    num_train_epochs=10,
    learning_rate=5e-5,
    max_length=1024,  # Ensure this matches the tokenizer max_length in preprocess_function
    # Add any additional configurations as needed
)
trainer = RewardTrainer(
    model=model,
    args=reward_config,
    tokenizer=tokenizer,
    train_dataset=dataset['train'],
    peft_config=peft_config,  # Convert LoraConfig to the expected format by get_peft_config if needed
    # eval_dataset can be added if you have validation data
)

trainer.train()
