import torch
from datasets import load_dataset
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer


def sft_trainer(dataset_name, base_model, output_dir):
    dataset = load_dataset(dataset_name, split='train')

    def preprocess_data(batch):
        prompts = [f"Generate a '{genre}' story titled '{title}'." for genre, title in
                   zip(batch['genre'], batch['title'])]
        return {'prompt': prompts, 'response': batch['story']}

    dataset = dataset.map(preprocess_data, batched=True, remove_columns=['id', 'title', 'genre', 'story'])

    base_model = base_model

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=4,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            'q_proj',
            'v_proj'
        ]
    )
    model = get_peft_model(model, peft_config)

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=10,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        optim="paged_adamw_32bit",
        save_strategy="epoch",
        logging_steps=100,
        logging_strategy="steps",
        learning_rate=2e-4,
        fp16=False,
        bf16=False,
        group_by_length=True,
        disable_tqdm=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        max_seq_length=4096,
        dataset_text_field="prompt",
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
    )

    trainer.train()


if __name__ == "__main__":
    output_dir = "./phi-2-finetuned-1k_stories_100_genre"
    base_model = "microsoft/phi-2"
    dataset_name = "FareedKhan/1k_stories_100_genre"
    sft_trainer(dataset_name, base_model, output_dir)
