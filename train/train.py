import os.path
import sys
import torch

sys.path.append(r'C:\Users\mirce\master\bert-moe')
sys.path.append(r'C:\Users\mirce\master\bert-moe\moe_bert')
sys.path.append(r'C:\Users\mirce\master\bert-moe\utils')
sys.path.append(r'C:\Users\mirce\master\bert-moe\train')
sys.path.append(r'C:\Users\mirce\master\bert-moe\moe_bert\moe_bert.py')
from moe_bert.moe_bert import MoeBert
from transformers import DataCollatorForLanguageModeling, BertTokenizer
import mlflow
from transformers import get_scheduler
from tqdm import tqdm
import time
from utils.losses import total_loss
from utils.utils import (set_seed,
                         prepare_dataset,
                         prepare_dataloaders,
                         get_model_config,
                         parse_args,
                         setup_tunneling,
                         get_chosen_experts)
import warnings
import mlflow.pytorch
import subprocess


def train():
    subprocess.Popen(["mlflow", "ui", "--port", "5000"])
    args = parse_args()
    if args.auth_token != '1':
        setup_tunneling(args)
    mlflow.set_experiment("pre-train-mlm-moe-bert")  # Create or set the experiment
    mlflow.start_run(log_system_metrics=True)
    mlflow.log_params(vars(args))
    model_config = get_model_config(args)
    set_seed(args.seed)
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', use_fast=True)
    mlm_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability)
    model = MoeBert(model_config).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    dataset = prepare_dataset(tokenizer, args)
    train_dataloader, eval_dataloader = prepare_dataloaders(dataset, mlm_collator, args)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none',
                                          label_smoothing=args.label_smoothing).to("cuda")
    num_training_steps = 365_000_000 * args.epochs // args.batch_size
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * num_training_steps,
        num_training_steps=num_training_steps
    )
    scaler = torch.amp.GradScaler()
    mlflow.log_text(model.__repr__(), "model_summary")
    mlflow.log_text("Training has started", "status")
    best_loss = float("inf")

    for epoch in range(args.epochs):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}", unit="batch")

        for batch in pbar:
            optimizer.zero_grad()
            input_ids, labels = batch["input_ids"].to("cuda"), batch["labels"].to("cuda")
            with torch.no_grad():
                mask_labels = (input_ids == tokenizer.mask_token_id).float()
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                output, routing_logits = model(input_ids)
                loss, pre_train_loss, balancing_loss = total_loss(output,
                                                                  labels,
                                                                  criterion,
                                                                  routing_logits,
                                                                  mask_labels,
                                                                  args)
            tokens_dispatch = get_chosen_experts(routing_logits, k=args.num_experts_per_token)
            for l in range(tokens_dispatch.shape[0]):
                mlflow.log_metric(f"expert_{l}_dispatch", tokens_dispatch[l], step=pbar.n)
            mlflow.log_metric("lr", optimizer.param_groups[0]['lr'], step=pbar.n)
            mlflow.log_metric("pre_train_loss", pre_train_loss.item(), step=pbar.n)
            mlflow.log_metric("balancing_loss", balancing_loss.item(), step=pbar.n)
            mlflow.log_metric("total_loss", loss.item(), step=pbar.n)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            # grad norm and param norm
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            param_norm = torch.norm(torch.cat([p.flatten() for p in model.parameters() if p.requires_grad]))
            mlflow.log_metric("grad_norm", grad_norm.item(), step=pbar.n)
            mlflow.log_metric("param_norm", param_norm.item(), step=pbar.n)
            # update progress bar
            train_loss += loss.item()
            pbar.set_postfix({"total_loss": loss.item()})
            if pbar.n > 0 and pbar.n % args.eval_steps == 0:
                avg_train_loss = train_loss / args.eval_steps
                mlflow.log_metric("train_loss", avg_train_loss, step=pbar.n)
                train_loss = 0.0
                # Evaluation
                model.eval()
                eval_loss = 0.0
                j = 0
                with torch.no_grad():
                    for batch_eval in eval_dataloader:
                        j += 1
                        input_ids, labels = batch_eval["input_ids"].to("cuda"), batch_eval["labels"].to("cuda")
                        mask_labels = (input_ids == tokenizer.mask_token_id).float()
                        output, routing_logits = model(input_ids)
                        loss, pre_train_loss, balancing_loss = total_loss(output, labels, criterion, routing_logits,
                                                                          mask_labels,
                                                                          args)
                        eval_loss += loss.item()

                avg_eval_loss = eval_loss / j
                mlflow.log_metric("eval_loss", avg_eval_loss, step=pbar.n)
                end_time = time.time()
                mlflow.log_metric("epoch duration", end_time - start_time, step=pbar.n)
                print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Eval Loss: {avg_eval_loss:.4f}")
                model.train()
                torch.cuda.empty_cache()
                if avg_eval_loss < best_loss:
                    best_loss = avg_eval_loss

                    storing_dir = os.path.join("mlruns",
                                               mlflow.active_run().info.experiment_id,
                                               mlflow.active_run().info.run_id,
                                               "artifacts")
                    mlflow.pytorch.log_model(model,
                                             registered_model_name='moe_bert',
                                             artifact_path="moe_bert_model",
                                             pip_requirements="requirements.txt")
                    torch.save(optimizer.state_dict(), os.path.join(storing_dir, "optimizer_state_dict.pth"))
                    torch.save(lr_scheduler.state_dict(), os.path.join(storing_dir, "lr_scheduler_state_dict.pth"))
                    torch.save(torch.get_rng_state(), os.path.join(storing_dir, "rng_state.pth"))
                    mlflow.log_artifact(os.path.join(storing_dir, "optimizer_state_dict.pth"))
                    mlflow.log_artifact(os.path.join(storing_dir, "lr_scheduler_state_dict.pth"))
                    mlflow.log_artifact(os.path.join(storing_dir, "rng_state.pth"))
                    mlflow.log_metric("best_eval_loss", best_loss, step=pbar.n)

    mlflow.end_run()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    train()
