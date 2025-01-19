# train_dpo.py
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from huggingface_hub import login
login()

# Load dataset
dataset_name = "vincentmin/eli5_rlhf_explainlikeim5"
train_dataset = load_dataset(dataset_name, split="train", data_files='data/rl/train-00000-of-00001.parquet')
print("Loaded train dataset:", train_dataset)

# Load model and tokenizer
model_name = "meta-llama/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Tokenize the dataset
def tokenize_function(examples):
    # Tokenize question and paired responses
    question_tokens = tokenizer(
        examples["question"], padding="max_length", truncation=True, max_length=256
    )
    response_j_tokens = tokenizer(
        examples["response_j"], padding="max_length", truncation=True, max_length=256
    )
    response_k_tokens = tokenizer(
        examples["response_k"], padding="max_length", truncation=True, max_length=256
    )
    
    # Combine question and responses into the format expected by DPO
    return {
        "input_ids": question_tokens["input_ids"],
        "attention_mask": question_tokens["attention_mask"],
        "response_j_ids": response_j_tokens["input_ids"],
        "response_j_mask": response_j_tokens["attention_mask"],
        "response_k_ids": response_k_tokens["input_ids"],
        "response_k_mask": response_k_tokens["attention_mask"],
    }

# Tokenize the entire dataset
train_dataset = train_dataset.map(tokenize_function, batched=True)
train_dataset.set_format(
    type="torch",
    columns=[
        "input_ids",
        "attention_mask",
        "response_j_ids",
        "response_j_mask",
        "response_k_ids",
        "response_k_mask",
    ],
)

# Define training arguments
training_args = DPOConfig(
    output_dir="./Llama-3.1-8B-DPO",
    beta=0.1,  # Regularization strength
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=1e-5,
    logging_dir="./logs",
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    evaluation_strategy="steps",
    eval_steps=500,
    warmup_steps=100,
    weight_decay=0.01,
    fp16=True,
    push_to_hub=False,
)

# Define DPO-specific configuration
dpo_config = DPOConfig(
    beta=0.1,  # Regularization strength
    output_dir="./Llama-3.1-8B-DPO",
)

# Initialize DPOTrainer
trainer = DPOTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
#     output_dir="./Llama-3.1-8B-DPO",
#     beta=0.1,  # Regularization strength
#     # training_args=dpo_config,
)

# Train the model
trainer.train()

# Save the model
trainer.save_model("./Llama-3.1-8B-DPO")
