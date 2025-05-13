import os
import requests
import torch
import pickle
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BlipProcessor, BlipForQuestionAnswering
from datasets import load_dataset

# ★ PEFT / LoRA imports
from peft import LoraConfig, get_peft_model

# 1) Load processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")

# 2) Define your LoRA config
peft_config = LoraConfig(
    r=8,                          # low-rank dimension
    lora_alpha=32,                 # scaling
    target_modules=['qkv', 'projection'],  # BLIP’s attention proj layers
    lora_dropout=0.05
)

# 3) Load the base model and wrap with LoRA adapters
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
model = get_peft_model(model, peft_config)

# 4) Move to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

torch.cuda.empty_cache()
torch.manual_seed(42)


class VQADataset(torch.utils.data.Dataset):
    """VQA (v2) dataset."""

    def __init__(self, dataset, processor, images_folder):
        self.dataset = dataset
        self.processor = processor
        self.images_folder = images_folder

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        question = self.dataset[idx]['question']
        answer   = self.dataset[idx]['answer']
        # assumes you passed in a base folder where all images live
        rel_path   = self.dataset[idx]['full_image_path']              # e.g. "abo-images-small/images/small\9d/9dfccb37.jpg"
        image_path = os.path.join(self.images_folder, rel_path.replace('\\','/'))
        image = Image.open(image_path).convert("RGB")

        encoding = self.processor(
            image, question,
            padding="max_length", truncation=True,
            return_tensors="pt"
        )
        labels = self.processor.tokenizer.encode(
            answer,
            max_length=8, pad_to_max_length=True,
            return_tensors='pt'
        )
        encoding["labels"] = labels

        # squeeze out the batch dimension
        return {k: v.squeeze() for k, v in encoding.items()}


# Load data
ds = load_dataset(
    "csv",
    data_files=r"C:\Users\subha\Desktop\vr proj\generated_questions\generated_questions\finalDataset.csv",
    split="train",
)

# optional: shuffle and then split 90/10
split = ds.shuffle(seed=42).train_test_split(test_size=0.1)
training_dataset = split["train"]
valid_dataset   = split["test"]

train_dataset = VQADataset(
    dataset=training_dataset,
    processor=processor,
    images_folder=r"C:\Users\subha\Desktop\vr proj"      
)
valid_dataset = VQADataset(
    dataset=valid_dataset,
    processor=processor,
    images_folder=r"C:\Users\subha\Desktop\vr proj"
)

batch_size = 12
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  pin_memory=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

# Only adapter parameters will be optimized:
optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5)

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

num_epochs = 100
patience   = 10
min_eval_loss = float("inf")
early_stop_counter = 0

scaler = torch.cuda.amp.GradScaler()
tracking_information = []

for epoch in range(num_epochs):
    # ——— TRAIN ———
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1} train"):
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.cuda.amp.autocast():
            outputs = model(**batch)
            loss = outputs.loss

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

    # ——— EVAL ———
    model.eval()
    eval_loss = 0.0
    for batch in tqdm(valid_dataloader, desc=f"Epoch {epoch+1} valid"):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad(), torch.cuda.amp.autocast():
            outputs = model(**batch)
            eval_loss += outputs.loss.item()

    train_loss /= len(train_dataloader)
    eval_loss  /= len(valid_dataloader)
    tracking_information.append((train_loss, eval_loss, optimizer.param_groups[0]["lr"]))
    print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, valid_loss={eval_loss:.4f}")

    # save best
    if eval_loss < min_eval_loss:
        min_eval_loss = eval_loss
        model.save_pretrained("Model/blip-lora-model")  # saves both base + adapters
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print("Early stopping.")
            break

    scheduler.step()

# dump loss history
with open("tracking_information.pkl", "wb") as f:
    pickle.dump(tracking_information, f)

print("Done!")
