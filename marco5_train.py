import torch
import os
import numpy as np
import random
import re
import time
from contextlib import nullcontext
from tokenizers import ByteLevelBPETokenizer

if os.getenv("FORCE_CPU", "0") == "1":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

from marco4_model import EdithLaravelModel, device, block_size, batch_size

# 1. Carregar Tokenizer
tokenizer = ByteLevelBPETokenizer("edith-laravel-grand/vocab.json", "edith-laravel-grand/merges.txt")
vocab_size = tokenizer.get_vocab_size()

seed = int(os.getenv("SEED", "1337"))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

# 2. Carregar Dataset de forma OTIMIZADA (Evita o "Morto")
def clean_text(text):
    text = text.replace("\x00", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def load_manifest_paths(data_root, manifest_path):
    paths = []
    if not manifest_path or not os.path.exists(manifest_path):
        return paths

    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            rel = line.strip()
            if not rel:
                continue
            abs_path = os.path.join(data_root, rel)
            if os.path.isfile(abs_path):
                paths.append(abs_path)
    return paths


def load_data_chunks():
    print("📚 Processando 10 milhões de tokens em modo econômico...")
    all_tokens = []
    files_seen = 0
    data_root = os.getenv("DATA_DIR", "data")
    manifest_path = os.getenv("DATA_MANIFEST", "")

    selected_files = load_manifest_paths(data_root, manifest_path)
    if selected_files:
        print(f"🧭 Usando manifesto curado: {manifest_path} ({len(selected_files)} arquivos)")

    if selected_files:
        file_iter = selected_files
    else:
        file_iter = []
        for root, _, files in os.walk(data_root):
            for file in files:
                if file.endswith((".php", ".md", ".json")):
                    file_iter.append(os.path.join(root, file))

    for file_path in file_iter:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = clean_text(f.read())
                tokens = tokenizer.encode(text).ids
                all_tokens.extend(tokens)
                files_seen += 1
        except:
            continue

    print(f"📂 Arquivos lidos: {files_seen}")
    return torch.tensor(all_tokens, dtype=torch.long)

# Agora carregamos os dados
full_data = load_data_chunks()
print(f"✅ Sucesso: {len(full_data)} tokens em memória.")
if len(full_data) < 10_000:
    raise RuntimeError("Dataset pequeno demais após tokenização; não há contexto suficiente para treinar com estabilidade.")

split_idx = int(len(full_data) * 0.95)
train_data = full_data[:split_idx]
val_data = full_data[split_idx:]
print(f"🧪 Split: treino={len(train_data)} tokens | validação={len(val_data)} tokens")

# 3. Inicializar Modelo com Hiperparâmetros seguros para baixa VRAM
if device == 'cuda':
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
else:
    total_vram_gb = 0

if device == 'cuda' and total_vram_gb <= 3:
    batch_size_safe = 4
    seq_len_safe = min(64, block_size)
    grad_accum_steps = int(os.getenv("GRAD_ACCUM", "8"))
else:
    batch_size_safe = min(16, batch_size)
    seq_len_safe = min(128, block_size)
    grad_accum_steps = int(os.getenv("GRAD_ACCUM", "4"))

print(f"⚙️ Modo seguro ativo -> batch={batch_size_safe}, seq_len={seq_len_safe}, grad_accum={grad_accum_steps}")

model = EdithLaravelModel(vocab_size).to(device)
base_lr = float(os.getenv("BASE_LR", "3e-4"))
min_lr = float(os.getenv("MIN_LR", "1e-5"))
warmup_steps = int(os.getenv("WARMUP_STEPS", "200"))
weight_decay = float(os.getenv("WEIGHT_DECAY", "0.1"))
max_grad_norm = float(os.getenv("MAX_GRAD_NORM", "1.0"))
eval_interval = int(os.getenv("EVAL_INTERVAL", "100"))
eval_iters = int(os.getenv("EVAL_ITERS", "20"))
step_sleep = float(os.getenv("STEP_SLEEP", "0"))

optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))
train_steps = int(os.getenv("TRAIN_STEPS", "1000"))
checkpoint_path = os.getenv("CHECKPOINT_PATH", "edith_laravel_senior_ckpt.pt")
checkpoint_every = int(os.getenv("CHECKPOINT_EVERY", "100"))
resume_training = os.getenv("RESUME", "1") == "1"
start_step = 0
best_val_loss = float('inf')


def get_lr(step):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, train_steps - warmup_steps)
    cosine = 0.5 * (1 + np.cos(np.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine

def save_checkpoint(step, loss_value=None):
    payload = {
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "loss": loss_value,
        "best_val_loss": best_val_loss,
        "batch_size_safe": batch_size_safe,
        "seq_len_safe": seq_len_safe,
        "grad_accum_steps": grad_accum_steps,
    }
    torch.save(payload, checkpoint_path)

if resume_training and os.path.exists(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scaler.load_state_dict(ckpt["scaler_state"])
    start_step = int(ckpt.get("step", 0))
    best_val_loss = float(ckpt.get("best_val_loss", float('inf')))
    print(f"♻️ Checkpoint carregado: retomando da iteração {start_step}")

def get_batch(data):
    ix = torch.randint(len(data) - seq_len_safe - 1, (batch_size_safe,))
    # Adicionamos o .to(device) aqui para mandar os dados para a GPU
    x = torch.stack([data[i:i+seq_len_safe] for i in ix]).to(device)
    y = torch.stack([data[i+1:i+seq_len_safe+1] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def estimate_loss(data_source, steps=20):
    model.eval()
    losses = []
    amp_ctx = torch.amp.autocast('cuda') if device == 'cuda' else nullcontext()
    for _ in range(steps):
        xb, yb = get_batch(data_source)
        with amp_ctx:
            _, loss = model(xb, yb)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if losses else float('inf')

# 4. Loop de Treino
print("🔥 Forjando a Edith (Versão Econômica)...")
try:
    for iter in range(start_step, train_steps):
        lr_now = get_lr(iter)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_now

        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0

        for _ in range(grad_accum_steps):
            xb, yb = get_batch(train_data)

            amp_ctx = torch.amp.autocast('cuda') if device == 'cuda' else nullcontext()
            with amp_ctx:
                _, loss = model(xb, yb)

            running_loss += loss.item()
            scaled_loss = loss / grad_accum_steps
            scaler.scale(scaled_loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        train_loss = running_loss / grad_accum_steps

        if iter % 50 == 0:
            print(f"Iteração {iter}: Train Loss = {train_loss:.4f} | LR = {lr_now:.6f}")

        if (iter + 1) % eval_interval == 0:
            val_source = val_data if len(val_data) > (seq_len_safe + 2) else train_data
            val_loss = estimate_loss(val_source, steps=eval_iters)
            print(f"📈 Validação {iter + 1}: Val Loss = {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'edith_laravel_senior_best.pt')
                print(f"🥇 Novo melhor modelo salvo! Val Loss = {best_val_loss:.4f}")

        if (iter + 1) % checkpoint_every == 0:
            save_checkpoint(iter + 1, train_loss)
            print(f"💾 Checkpoint salvo na iteração {iter + 1}")

        if step_sleep > 0:
            time.sleep(step_sleep)
except KeyboardInterrupt:
    print("⏸️ Treino interrompido manualmente. Salvando checkpoint...")
    save_checkpoint(iter if 'iter' in locals() else start_step)
    raise
except Exception:
    print("🚨 Falha durante treino. Salvando checkpoint de emergência...")
    save_checkpoint(iter if 'iter' in locals() else start_step)
    raise

torch.save(model.state_dict(), 'edith_laravel_senior.pt')
print("🏆 MODELO SALVO!")
