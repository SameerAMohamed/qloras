# Configurable constants
import huggingface_hub

CUDA_VISIBLE_DEVICE = "0"
MODEL_HFACE_TAG = "microsoft/phi-2"
DATASET_HFACE_TAG = "corbt/all-recipes"
OUTPUT_DIR = '../outputs'
MODEL_SAVE_PATH = f'./{MODEL_HFACE_TAG}_{DATASET_HFACE_TAG}_recipes_qlora'
TOKENIZER_SAVE_PATH = f'./{MODEL_HFACE_TAG}_{DATASET_HFACE_TAG}_qlora_tokenizer'
MODEL_SAVE_PATH = '/gcs/your-cloud-project-sameermohamed1-bucket/'
TOKENIZER_SAVE_PATH = '/gcs/your-cloud-project-sameermohamed1-bucket/'
EVALUATION_STRATEGY = "steps"
EVAL_STEPS = 10
MAX_STEPS = 50
WARMUP_STEPS = 10
LEARNING_RATE = 2e-4
TRAIN_SPLIT_PERCENT = 90  # 90% of the data will be used for training
DATA_FILE_PATH = '/gcs/your-cloud-project-sameermohamed1-bucket/trainer/processed_data.pkl'  # Path to save/load the processed data

from huggingface_hub import login
import pickle
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import bitsandbytes as bnb
from datasets import load_dataset, DatasetDict
from peft import LoraConfig, get_peft_model
import wandb



# Set environment variables for Hugging Face and wandb and login
#wandb.login()
#login()

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICE

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    MODEL_HFACE_TAG,
    load_in_4bit=True,
    device_map='auto'
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_HFACE_TAG)
tokenizer.pad_token = tokenizer.eos_token

# Freeze model parameters and cast small params to fp32
for param in model.parameters():
    param.requires_grad = False
    if param.ndim == 1:
        param.data = param.data.to(torch.float32)

# Enable gradient checkpointing and input grad requirement
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

# Custom class to cast output of the lm_head to float32
class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)

model.lm_head = CastOutputToFloat(model.lm_head)

# Add PEFT model modifications
config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(model, config)

# Check if the processed data exists
if not os.path.exists(DATA_FILE_PATH):
    # Load dataset and create train/validation split
    data = load_dataset(DATASET_HFACE_TAG)
    train_size = int(len(data['train']) * TRAIN_SPLIT_PERCENT / 100)
    validation_size = len(data['train']) - train_size
    data = DatasetDict({
        'train': data['train'].shuffle(seed=42).select(range(train_size)),
        'validation': data['train'].shuffle(seed=42).select(range(train_size, len(data['train'])))
    })

    def merge_columns(example):
        example["prediction"] = "This is an example of a recipe: \n\n" + example["input"]
        return example

    data['train'] = data['train'].map(merge_columns)
    data['validation'] = data['validation'].map(merge_columns)
    data = data.map(lambda samples: tokenizer(samples['prediction']), batched=True)

    # Save processed data
    with open(DATA_FILE_PATH, 'wb') as f:
        pickle.dump(data, f)
else:
    # Load processed data
    with open(DATA_FILE_PATH, 'rb') as f:
        data = pickle.load(f)

# Trainer setup
trainer = Trainer(
    model=model,
    train_dataset=data['train'],
#    eval_dataset=data['validation'],  # Set validation dataset
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=1,
        output_dir=OUTPUT_DIR,
#        evaluation_strategy=EVALUATION_STRATEGY,
#        eval_steps=EVAL_STEPS,
        save_strategy="no",  # To avoid saving the model multiple times unnecessarily
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
)

# Disable caching for training
model.config.use_cache = False

# Train the model and evaluate
trainer.train()

# Save model and tokenizer
trainer.save_model(MODEL_SAVE_PATH)
model.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(TOKENIZER_SAVE_PATH)
