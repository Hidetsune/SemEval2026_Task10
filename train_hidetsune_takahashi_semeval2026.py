import json
import logging
import os
import random

import datasets
import evaluate
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_scheduler,
)
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available else "cpu")
print(f"Using device {device}")




logger = get_logger(__name__)

def main():
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()


    if seed: set_seed(seed)


    #Loading Data
    data_files = {}
    data_files["train"] = train_file
    data_files["validation"] = validation_file
    extension = (train_file if train_file is not None else validation_file).split(".")[-1] #extract extention
    raw_datasets = load_dataset(extension, data_files=data_files)


    #obtain the number of labels
    label_list = raw_datasets["train"].unique("label")
    label_list.sort()
    num_labels = len(label_list)

    #load model and tokenizer
    config = AutoConfig.from_pretrained(
        path_to_weight,
        num_labels=num_labels,
    )
    tokenizer = AutoTokenizer.from_pretrained(path_to_weight)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token #for gpt seriese
    elif tokenizer.cls_token is not None: #for bert seriese
        tokenizer.pad_token = tokenizer.cls_token
    else:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    config.pad_token_id = tokenizer.pad_token_id
    model = AutoModelForSequenceClassification.from_pretrained(
        path_to_weight,
        config=config,
        ignore_mismatched_sizes=ignore_mismatched_sizes,
    )
    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    # Freeze model's core backbone (for sample size < 300)
    if freeze_model_backbone:
        print("Attempting to freeze model backbone...")
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        # Display trainable (non-frozen) & non-trainable (frozen) params
        trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
        frozen_params = [n for n, p in model.named_parameters() if n not in trainable_params]
        print(f"Trainable params: {trainable_params}")
        print(f"Frozen params: {frozen_params}")


    non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
    if len(non_label_column_names) >= 2:
        sentence1_key, sentence2_key = non_label_column_names[:2]
    else:
        sentence1_key, sentence2_key = non_label_column_names[0], None



    def preprocess_function(examples):
        texts = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*texts, padding=False, max_length=max_length, truncation=True)
        if "label" in examples: result["labels"] = examples["label"]
        return result

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]

    #WC
    label_counts = Counter(train_dataset["labels"])
    total = sum(label_counts.values())
    weights = [total / label_counts[i] for i in range(num_labels)]
    class_weights = torch.tensor(weights).to(accelerator.device)

    # logging a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")


    if accelerator.mixed_precision == "fp8":
        pad_to_multiple_of = 16
    elif accelerator.mixed_precision != "no":
        pad_to_multiple_of = 8
    else:
        pad_to_multiple_of = None
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=pad_to_multiple_of)

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=train_batch
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=32) #CHANGE HERE


    no_decay = ["bias", "LayerNorm.weight"] #Do not place weight decay for these two weights
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)

    train_steps = len(train_dataloader) * train_epochs

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=train_steps,
    )

    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    metric = evaluate.load(eval_metric)

    # Training
    total_batch_size = train_batch * accelerator.num_processes

    logger.info("Training Started")
    # Only show the progress bar once on each machine.
    completed_steps = 0
    starting_epoch = 0

    print(starting_epoch, train_epochs)

    early_stopping_trigger = False  # initialize trigger for potential early stopping
    patience_counter = 0  # initialize patience counter
    best_metric = -float("inf")  # initialize best score (-infinite)
    best_epoch = None
    total_loss = 0.0
    for epoch in range(starting_epoch, train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            logits = outputs.logits
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fct(logits, batch["labels"])
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            completed_steps += 1

        # Set the model to eval mode
        model.eval()
        metric = evaluate.load(eval_metric)
        samples_seen = 0
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad(): # set to no grad mode
                outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = accelerator.gather((predictions, batch["labels"]))
            if accelerator.num_processes > 1:
                if step == len(eval_dataloader) - 1:
                    predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                    references = references[: len(eval_dataloader.dataset) - samples_seen]
                else:
                    samples_seen += references.shape[0]

            predictions = predictions.detach().cpu().numpy().reshape(-1)
            references = references.detach().cpu().numpy().reshape(-1)

            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        if eval_metric in ["precision", "recall", "f1"] and num_labels>=3:
            results  = metric.compute(average="macro")
        else:
            results = metric.compute()
        logger.info(f"epoch {epoch}: {results}")

        current_score = results[eval_metric]
        if current_score > best_metric:
            patience_counter = 0
            best_epoch = epoch
            # save the current best model
            if output_dir is not None:
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    os.path.join(output_dir, "best_model_checkpoint"),
                    is_main_process=accelerator.is_main_process, save_function=accelerator.save
                )
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(os.path.join(epoch_output_dir, "best_model_checkpoint"))

            best_metric = current_score  # update best score
        else:
            patience_counter += 1
            accelerator.print(f"No improvement. patience={patience_counter}/{patience}")
        total_loss += loss.detach().item()
        if patience_counter >= patience:
            accelerator.print("Early stopping triggered.")
            early_stopping_trigger = True

            accelerator.log(
                {
                    "accuracy": current_score,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        if early_stopping_trigger:
            break


        output_subdir = f"epoch_{epoch}"
        if output_dir is not None:
            epoch_output_dir = os.path.join(output_dir, output_subdir)
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            epoch_output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(output_dir)
    #out of loop


    accelerator.end_training()
    logger.info(f"BEST-performing epoch num: {best_epoch}")

    if output_dir is not None:
        all_results = {}
        with open(os.path.join(output_dir, "all_results.json"), "w") as f:
            json.dump(all_results, f)


if __name__ == "__main__":
    train_file = "" #CHANGE HERE
    validation_file = "" #CHANGE HERE
    output_dir = "" #CHANGE HERE
    path_to_weight = "" #CHANGE HERE
    train_epochs = 10
    patience = 2
    max_length = 128
    lr = 2e-5
    train_batch = 16
    weight_decay = 0.01
    warmup_steps = 50
    seed = 42
    eval_metric = "f1"
    ignore_mismatched_sizes = True
    freeze_model_backbone = False #could be set True for lower resource situations

    main()




