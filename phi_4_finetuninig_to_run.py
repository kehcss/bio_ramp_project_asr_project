
# Requirments; only run this if you don't have the packages installed
# !pip install requests
# !pip install torch
# !pip install sacrebleu
# !pip install accelerate
# !pip install datasets
# !pip install transformers
# !pip install scipy torchvision
# !pip install peft
# !pip install backoff
# !pip install soundfile
# !pip install librosa

"""## Load Librarys"""

"""
finetune Phi-4-multimodal-instruct on an speech task

scipy==1.15.1
peft==0.13.2
backoff==2.2.1
transformers==4.46.1
accelerate==1.3.0
"""

import argparse
import json
import os
from pathlib import Path
import requests
import tarfile
import shutil
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import warnings
import zipfile

import torch
import sacrebleu
from accelerate import Accelerator
from accelerate.utils import gather_object
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BatchFeature,
    Trainer,
    TrainingArguments,
    StoppingCriteria,
    StoppingCriteriaList,
)

"""## Download testsets from afrivox"""

def download_and_extract_tar_gz(url, extract_to):
    """Download and extract a tar.gz file."""
    tar_gz_path = os.path.join(extract_to, 'audio.tar.gz')

    # Create directory if it does not exist
    os.makedirs(extract_to, exist_ok=True)

    # Download the tar.gz file
    response = requests.get(url, stream=True, allow_redirects=True)
    response.raise_for_status()
    os.makedirs(extract_to, exist_ok=True)
    with open(tar_gz_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    # Extract the tar.gz file
    with tarfile.open(tar_gz_path, 'r:gz') as tar_ref:
        tar_ref.extractall(path=extract_to)

    # # Remove the tar.gz file after extraction
    # os.remove(tar_gz_path)

# URL for the tar.gz file
tar_gz_url = 'https://drive.usercontent.google.com/download?id=11gqUWtWMeYTg2QnUAK4MHfPTGsXd_iXI&export=download&authuser=0&confirm=t&uuid=3e379af4-29c2-4252-8e8a-d8d9e6402a4e&at=AKSUxGM2VIa7Gk3vXVL_x18W8cIo:1760156638880'

# Directory to extract the contents
extract_to = 'data_folder/afrivox'


# Download and extract
download_and_extract_tar_gz(tar_gz_url, extract_to)

"""## Clean up testset"""

# Load the test csv file
test_csv = pd.read_csv('data_folder/afrivox/afrivox_transcribe_metadata.csv')

# # Select rows with "language" == "hausa", "igbo", or "yoruba"
# test_csv = test_csv[(test_csv['language'] == 'hausa') |
#                     (test_csv['language'] == 'igbo') |
#                     (test_csv['language'] == 'yoruba')]

# Rename columns to match the expected structure
test_csv.rename(columns={'speaker_id': 'client_id',
                         'audio_path': 'path',
                         'transcription': 'sentence'}, inplace=True)

# Modify the path to point to the correct location
test_csv['path'] = test_csv['path'].apply(lambda x: os.path.join('data_folder/afrivox', x))

# This function checks if the files in the dataset exist and removes the rows with nonexistent files
def remove_nonexistent_files(df):
    df = df[df['path'].apply(lambda x: os.path.exists(x))]
    return df

# Remove nonexistent files from the DataFrame
test_csv = remove_nonexistent_files(test_csv)

# Save the cleaned DataFrame back to the CSV file
test_csv.to_csv('data_folder/afrivox/test.csv', index=False)
print(f"Cleaned and saved the test CSV file.")

"""## Download train and val sets from naijavoice as multi zip archieves"""

def download_and_extract_zip(url, extract_to):
    """Download and extract a zip file."""
    zip_path = os.path.join(extract_to, 'naijavoices_compressed.zip')

    # Download the zip file
    response = requests.get(url, stream=True, allow_redirects=True)
    response.raise_for_status()
    os.makedirs(extract_to, exist_ok=True)
    with open(zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    # Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(path=extract_to)

    # Remove the zip file after extraction
    os.remove(zip_path)

# URL for the zip file
zip_url = 'https://uc99d15b7380b3e8bf16642b24ff.dl.dropboxusercontent.com/cd/0/get/CzDEd4_-HUZFgTMh-Oe3l1MAEhthpWi529ZWHvL8e_lw3HrPAXXwwT4B6_L0klQDLucpInEmC2vp7WgtYIELYLKm5A-5bSjjZHe1cxyP5fxmlJ9c8pL16TJQbxHIv6bDWt9owO8MKQ7HgHliWQ1elCxKJRt2zu4P1zlMHzvZwSWYZQ/file?_download_id=80777141644934137824410352493734853424771219785348959773005169915&_log_download_success=1#'

# Directory to extract the contents
extract_to = 'data_folder/afrivox'

# Download and extract
download_and_extract_zip(zip_url, extract_to)

"""## Edit audio path and discard audio files not in directory"""

train_val_dataset = pd.read_csv('data_folder/afrivox/nv_train_subset_real.csv')
#rename speaker_id to client_id
train_val_dataset.rename(columns={'speaker_id': 'client_id'}, inplace=True)
train_val_dataset.rename(columns={'audio_path': 'path'}, inplace=True)
train_val_dataset.rename(columns={'transcription': 'sentence'}, inplace=True)

# Modify the path to point to the correct location
train_val_dataset['path'] = train_val_dataset['path'].apply(lambda x: os.path.join('data_folder/afrivox', x))

# Function to mark file availability in the dataset
# This function checks if the files in the dataset exist and marks their availability
def mark_file_availability(df):
    """Mark file availability in the DataFrame."""
    df['Available'] = df['path'].apply(lambda x: 1 if os.path.exists(x) else 0)
    return df

# Update the train_val_dataset DataFrame
train_val_dataset = mark_file_availability(train_val_dataset)

"""## Tran/Val Split"""

# Use the existing DataFrame directly
df = train_val_dataset

# Split the dataset into train and validation sets
train_dataset = df.sample(frac=0.8, random_state=42)
val_dataset = df.drop(train_dataset.index)

# Save the datasets to CSV files
train_dataset.to_csv('data_folder/afrivox/train.csv', index=False)
val_dataset.to_csv('data_folder/afrivox/validated.csv', index=False)

"""## Training Pipeline code"""

# Ignore the warning about librosa
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")



INSTSRUCTION = {
    "en_zh-CN": "Translate the audio to Mandarin.",
    "en_id": "Translate the audio to Indonesian.",
    "en_sl": "Translate the audio to Slovenian.",
    "id_en": "Translate the audio from Indonesian to English.",
    "sl_en": "Translate the audio from Slovenian to English.",
    "zh_en": "Translate the audio from Mandarin to English.",
    "sl_en": "Translate the audio from Slovenian to English.",
}
TOKENIZER = {
    "en_zh-CN": "zh",
    "en_ja": "ja-mecab",
}
ANSWER_SUFFIX = "<|end|><|endoftext|>"
_IGNORE_INDEX = -100
_TRAIN_SIZE = 50000
_EVAL_SIZE = 200

class MultipleTokenBatchStoppingCriteria(StoppingCriteria):
    """Stopping criteria capable of receiving multiple stop-tokens and handling batched inputs."""

    def __init__(self, stop_tokens: torch.LongTensor, batch_size: int = 1) -> None:
        """Initialize the multiple token batch stopping criteria.

        Args:
            stop_tokens: Stop-tokens.
            batch_size: Batch size.

        """

        self.stop_tokens = stop_tokens
        self.max_stop_tokens = stop_tokens.shape[-1]
        self.stop_tokens_idx = torch.zeros(batch_size, dtype=torch.long, device=stop_tokens.device)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Only gather the maximum number of inputs compatible with stop tokens
        # and checks whether generated inputs are equal to `stop_tokens`
        generated_inputs = torch.eq(input_ids[:, -self.max_stop_tokens :].unsqueeze(1), self.stop_tokens)
        equal_generated_inputs = torch.all(generated_inputs, dim=2)

        # Mark the position where a stop token has been produced for each input in the batch,
        # but only if the corresponding entry is not already set
        sequence_idx = torch.any(equal_generated_inputs, dim=1)
        sequence_set_mask = self.stop_tokens_idx == 0
        self.stop_tokens_idx[sequence_idx & sequence_set_mask] = input_ids.shape[-1]

        return torch.all(self.stop_tokens_idx)

class CustomDataset(Dataset):
    def __init__(self, processor, data_dir, split, lang="en_zh-CN", rank=0, world_size=1):
        # Load the custom dataset from CSV files
        file_path = os.path.join(data_dir, f"{split}.csv")
        self.data = pd.read_csv(file_path)

        self.training = "train" in split
        self.processor = processor
        self.instruction = INSTSRUCTION[lang]

        # For distributed training, shard the dataset if needed
        if world_size > 1:
            self.data = self.data.iloc[rank::world_size]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data.iloc[idx]
        user_message = {
            'role': 'user',
            'content': '<|audio_1|>\n' + self.instruction,
        }
        prompt = self.processor.tokenizer.apply_chat_template(
            [user_message], tokenize=False, add_generation_prompt=True
        )
        audio_path = data["path"]
        # Suppress FutureWarning from librosa


        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
            audio_array, sampling_rate = librosa.load(audio_path, sr=None)


        inputs = self.processor(
            text=prompt,
            audios=[(audio_array, sampling_rate)],
            return_tensors='pt'
        )

        answer = f"{data['sentence']}{ANSWER_SUFFIX}"
        answer_ids = self.processor.tokenizer(answer, return_tensors='pt').input_ids
        if self.training:
            input_ids = torch.cat([inputs.input_ids, answer_ids], dim=1)
            labels = torch.full_like(input_ids, _IGNORE_INDEX)
            labels[:, -answer_ids.shape[1]:] = answer_ids
        else:
            input_ids = inputs.input_ids
            labels = answer_ids

        return {
            'input_ids': input_ids,
            'labels': labels,
            'input_audio_embeds': inputs.input_audio_embeds,
            'audio_embed_sizes': inputs.audio_embed_sizes,
        }


def pad_sequence(sequences, padding_side='right', padding_value=0):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output


def cat_with_pad(tensors, dim, padding_value=0):
    """
    cat along dim, while pad to max for all other dims
    """
    ndim = tensors[0].dim()
    assert all(
        t.dim() == ndim for t in tensors[1:]
    ), 'All tensors must have the same number of dimensions'

    out_size = [max(t.shape[i] for t in tensors) for i in range(ndim)]
    out_size[dim] = sum(t.shape[dim] for t in tensors)
    output = tensors[0].new_full(out_size, padding_value)

    index = 0
    for t in tensors:
        # Create a slice list where every dimension except dim is full slice
        slices = [slice(0, t.shape[d]) for d in range(ndim)]
        # Update only the concat dimension slice
        slices[dim] = slice(index, index + t.shape[dim])

        output[slices] = t
        index += t.shape[dim]

    return output


def covost_collate_fn(batch):
    input_ids_list = []
    labels_list = []
    input_audio_embeds_list = []
    audio_embed_sizes_list = []
    audio_attention_mask_list = []
    for inputs in batch:
        input_ids_list.append(inputs['input_ids'][0])
        labels_list.append(inputs['labels'][0])
        input_audio_embeds_list.append(inputs['input_audio_embeds'])
        audio_embed_sizes_list.append(inputs['audio_embed_sizes'])
        audio_attention_mask_list.append(
            inputs['input_audio_embeds'].new_full((inputs['input_audio_embeds'].size(1),), True, dtype=torch.bool)
        )

    try:
        input_ids = pad_sequence(input_ids_list, padding_side='left', padding_value=0)
        labels = pad_sequence(labels_list, padding_side='left', padding_value=0)
        audio_attention_mask = (
            pad_sequence(audio_attention_mask_list, padding_side='right', padding_value=False)
            if len(audio_attention_mask_list) > 1
            else None
        )
    except Exception as e:
        print(e)
        print(input_ids_list)
        print(labels_list)
        raise
    attention_mask = (input_ids != 0).long()
    input_audio_embeds = cat_with_pad(input_audio_embeds_list, dim=0)
    audio_embed_sizes = torch.cat(audio_embed_sizes_list)

    return BatchFeature(
        {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'input_audio_embeds': input_audio_embeds,
            'audio_embed_sizes': audio_embed_sizes,
            'audio_attention_mask': audio_attention_mask,
            'input_mode': 2,  # speech mode
        }
    )



def create_model(model_name_or_path, use_flash_attention=False):
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16 if use_flash_attention else torch.float32,
        _attn_implementation='flash_attention_2' if use_flash_attention else 'sdpa',
        trust_remote_code=True,
    ).to('cuda')

    return model


@torch.no_grad()
def evaluate(
    model, processor, eval_dataset, save_path=None, disable_tqdm=False, eval_batch_size=1
):
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    model.eval()
    all_generated_texts = []
    all_labels = []

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        collate_fn=covost_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=8,
        prefetch_factor=2,
        pin_memory=True,
    )
    stop_tokens = ["<|end|>", processor.tokenizer.eos_token]
    stop_tokens_ids = processor.tokenizer(stop_tokens, add_special_tokens=False, padding="longest", return_tensors="pt")["input_ids"]
    stop_tokens_ids = stop_tokens_ids.to(f'cuda:{local_rank}')

    for inputs in tqdm(
        eval_dataloader, disable=(rank != 0) or disable_tqdm, desc='running eval'
    ):
        stopping_criteria=StoppingCriteriaList([MultipleTokenBatchStoppingCriteria(stop_tokens_ids, batch_size=inputs.input_ids.size(0))])
        inputs = inputs.to(f'cuda:{local_rank}')
        generated_ids = model.generate(
            **inputs, eos_token_id=processor.tokenizer.eos_token_id, max_new_tokens=64,
            stopping_criteria=stopping_criteria,
            num_logits_to_keep=64  # Set an appropriate value
        )

        stop_tokens_idx = stopping_criteria[0].stop_tokens_idx.reshape(inputs.input_ids.size(0), -1)[:, 0]

        stop_tokens_idx = torch.where(
            stop_tokens_idx > 0,
            stop_tokens_idx - stop_tokens_ids.shape[-1],
            generated_ids.shape[-1],
        )
        generated_text = [
            processor.decode(_pred_ids[inputs["input_ids"].shape[1] : _stop_tokens_idx], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for _pred_ids, _stop_tokens_idx in zip(generated_ids, stop_tokens_idx)
        ]
        all_generated_texts.extend(generated_text)
        labels = [processor.decode(_label_ids[_label_ids != 0]).removesuffix(ANSWER_SUFFIX) for _label_ids in inputs["labels"]]
        all_labels.extend(labels)

    all_generated_texts = gather_object(all_generated_texts)
    all_labels = gather_object(all_labels)

    if rank == 0:
        assert len(all_generated_texts) == len(all_labels)
        bleu = sacrebleu.corpus_bleu(all_generated_texts, [all_labels])
        print(bleu)
        if save_path:
            with open(save_path, 'w') as f:
                save_dict = {
                    'all_generated_texts': all_generated_texts,
                    'all_labels': all_labels,
                    'score': bleu.score,
                }
                json.dump(save_dict, f)

        return bleu.score
    return None


### EVALUATION AND THEN TRAINING
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        default='microsoft/Phi-4-multimodal-instruct',
        help='Model name or path to load from',
    )
    parser.add_argument(
        "--common_voice_dir",
        type=str,
        default="data_folder/afrivox",
        help="Custom dataset directory",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en_sl",
        help="Language pair for translation.",
    )
    parser.add_argument('--use_flash_attention', action='store_true', help='Use Flash Attention')
    parser.add_argument('--output_dir', type=str, default='./output/', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument(
        '--batch_size_per_gpu',
        type=int,
        default=1,
        help='Batch size per GPU (adjust this to fit in GPU memory)',
    )
    #parser.add_argument('--num_train_epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=4.0e-5, help='Learning rate')
    parser.add_argument('--wd', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--no-tqdm', dest='tqdm', action='store_false', help='Disable tqdm')
    args = parser.parse_args(args=[])

    accelerator = Accelerator()

    with accelerator.local_main_process_first():
        processor = AutoProcessor.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
        )
        model = create_model(
            args.model_name_or_path,
            use_flash_attention=args.use_flash_attention,
        )

    model.set_lora_adapter('speech')

    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    train_dataset = CustomDataset(
        processor,
        data_dir=args.common_voice_dir,
        split='train',
        lang=args.lang,
    )

    eval_dataset = CustomDataset(
        processor,
        data_dir=args.common_voice_dir,
        split='test',
        lang=args.lang,
        rank=rank,
        world_size=world_size,
    )



    num_gpus = accelerator.num_processes
    print(f'training on {num_gpus} GPUs')
    assert (
        args.batch_size % (num_gpus * args.batch_size_per_gpu) == 0
    ), 'Batch size must be divisible by the number of GPUs'
    gradient_accumulation_steps = args.batch_size // (num_gpus * args.batch_size_per_gpu)

    if args.use_flash_attention:
        fp16 = False
        bf16 = True
    else:
        fp16 = True
        bf16 = False

    # Training arguments
    training_args = TrainingArguments(
        #num_train_epochs=args.num_train_epochs,
        max_steps=300,
        per_device_train_batch_size=args.batch_size_per_gpu,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim='adamw_torch',
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-7,
        learning_rate=args.learning_rate,
        weight_decay=args.wd,
        max_grad_norm=1.0,
        lr_scheduler_type='linear',
        warmup_steps=50,
        logging_steps=10,
        output_dir=args.output_dir,
        save_strategy='no',
        save_total_limit=10,
        save_only_model=True,
        bf16=bf16,
        fp16=fp16,
        remove_unused_columns=False,
        report_to='none',
        deepspeed=None,
        disable_tqdm=not args.tqdm,
        dataloader_num_workers=4,
        ddp_find_unused_parameters=True,
    )


    # eval before fine-tuning
    out_path = Path(training_args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Clear GPU memory before evaluation
    torch.cuda.empty_cache()

    score = evaluate(
        model,
        processor,
        eval_dataset,
        save_path=out_path / 'eval_before.json',
        disable_tqdm=not args.tqdm,
        eval_batch_size=args.batch_size_per_gpu,
    )
    if accelerator.is_main_process:
        print(f'BLEU Score before finetuning: {score}')

    # clear GPU memory before training
    torch.cuda.empty_cache()

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=covost_collate_fn,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model()
    if accelerator.is_main_process:
        processor.save_pretrained(training_args.output_dir)
    accelerator.wait_for_everyone()



    # eval after fine-tuning (load saved checkpoint)
    # first try to clear GPU memory
    del model
    del trainer
    __import__('gc').collect()
    torch.cuda.empty_cache()

    # reload the model for inference
    model = AutoModelForCausalLM.from_pretrained(
        training_args.output_dir,
        torch_dtype=torch.bfloat16 if args.use_flash_attention else torch.float32,
        trust_remote_code=True,
        _attn_implementation='flash_attention_2' if args.use_flash_attention else 'sdpa',
    ).to('cuda')

    # Clear GPU memory before evaluation
    torch.cuda.empty_cache()

    # Disable gradient checkpointing during evaluation
    model.gradient_checkpointing_disable()

    # Evaluate before fine-tuning
    score = evaluate(
        model,
        processor,
        eval_dataset,
        save_path=out_path / 'eval_after.json',
        disable_tqdm=not args.tqdm,
        eval_batch_size=args.batch_size_per_gpu,
    )
    print(f'BLEU Score before finetuning: {score}')





    # # Skip evaluation and proceed to fine-tuning
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     data_collator=covost_collate_fn,
    #     train_dataset=train_dataset,
    # )

    # trainer.train()
    # trainer.save_model()
    # if accelerator.is_main_process:
    #     processor.save_pretrained(training_args.output_dir)
    # accelerator.wait_for_everyone()


if __name__ == '__main__':
    main()