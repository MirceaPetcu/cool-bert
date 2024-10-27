import sys
import torch
from IPython import get_ipython

sys.path.append(r'C:\Users\mirce\master\bert-moe')
sys.path.append(r'C:\Users\mirce\master\bert-moe\moe_bert')
sys.path.append(r'C:\Users\mirce\master\bert-moe\utils')
from moe_bert.moe_bert import MoeBert
from transformers import DataCollatorForLanguageModeling, BertTokenizer
import mlflow
from transformers import get_scheduler
from tqdm import tqdm
import time
from utils.losses import total_loss
from utils.utils import set_seed, prepare_dataset, prepare_dataloaders, get_model_config, parse_args
import warnings
import mlflow.pytorch
import subprocess


def setup_mlflow(args):
    # IMP: please create a auth token from https://dashboard.ngrok.com/auth by creating an account.
    # the below auth ticket will not work for anyone re-running the notebook.

    from pyngrok import ngrok
    from getpass import getpass

    # Terminate open tunnels if exist
    ngrok.kill()

    # Setting the authtoken (optional)
    # Get your authtoken from https://dashboard.ngrok.com/auth
    NGROK_AUTH_TOKEN = args.auth_token
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)

    # Open an HTTPs tunnel on port 5000 for http://localhost:5000
    ngrok_tunnel = ngrok.connect(addr="5000", proto="http", bind_tls=True)
    print("MLflow Tracking UI:", ngrok_tunnel.public_url)


def train():
    subprocess.Popen(["mlflow", "ui", "--port", "5000"])
    # mlflow.pytorch.autolog()
    args = parse_args()
    setup_mlflow(args)
    # mlflow.set_tracking_uri('https://dagshub.com/MirceaPetcu/MLflow-integration.mlflow')  # Set the tracking URI
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
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='mean',
                                          label_smoothing=args.label_smoothing).to("cuda")
    num_training_steps = 365000000 * args.epochs // args.batch_size
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
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                output, routing_logits = model(input_ids)
                loss, pre_train_loss, balancing_loss = total_loss(output, labels, criterion, routing_logits, args)
            mlflow.log_metric("pre_train_loss", pre_train_loss.item(), step=epoch)
            mlflow.log_metric("balancing_loss", balancing_loss.item(), step=epoch)
            mlflow.log_metric("total_loss", loss.item(), step=epoch)
            # optimizer and lr_scheduler step
            # loss.backward()
            # optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            # grad norm and param norm
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            param_norm = torch.norm(torch.cat([p.flatten() for p in model.parameters() if p.requires_grad]))
            mlflow.log_metric("grad_norm", grad_norm.item(), step=epoch)
            mlflow.log_metric("param_norm", param_norm.item(), step=epoch)
            # update progress bar
            train_loss += loss.item()
            pbar.set_postfix({"total_loss": loss.item()})
            if pbar.n > 0 and pbar.n % args.eval_steps == 0:
                avg_train_loss = train_loss / pbar.n
                mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
                train_loss = 0.0
                # Evaluation
                model.eval()
                eval_loss = 0.0
                j = 0
                with torch.no_grad():
                    for batch_eval in eval_dataloader:
                        j += 1
                        input_ids, labels = batch_eval["input_ids"].to("cuda"), batch_eval["labels"].to("cuda")
                        output, routing_logits = model(input_ids)
                        loss, pre_train_loss, balancing_loss = total_loss(output, labels, criterion, routing_logits,
                                                                          args)
                        eval_loss += loss.item()

                avg_eval_loss = eval_loss / j
                mlflow.log_metric("eval_loss", avg_eval_loss, step=epoch)
                end_time = time.time()
                mlflow.log_metric("epoch duration", end_time - start_time, step=epoch)
                print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Eval Loss: {avg_eval_loss:.4f}")
                model.train()
                torch.cuda.empty_cache()
                if avg_eval_loss < best_loss:
                    best_loss = avg_eval_loss
                    mlflow.pytorch.log_model(model, model_name="moe_bert", registered_model_name='moe_bert',
                                             pip_requirements="requirements.txt")
                    mlflow.log_metric("best_eval_loss", best_loss, step=epoch)
    mlflow.end_run()


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    train()
