import argparse
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from amadeus.models.codec.seanet import SEANetEncoder
from amadeus.models.lm.transformer import SemanticTokenLM
from amadeus.models.lm.conditioning import TextConditioningEncoder
from amadeus.models.rvq.quantizer import ResidualVectorQuantizer
from amadeus.data.dataset import MusicAudioDataset, DataLoader
from amadeus.data.analysis.frontend import AudioAnalysisFrontend
from amadeus.pipeline.checkpoint import save_model_checkpoint, load_model_checkpoint

def train_lm(data_dir="./dataset", epochs=2, batch_size=2, lr=1.5e-4, save_dir="./checkpoints", device="cpu", compile_model=False, grad_accum=1, resume=False):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")
    use_cuda = device.type == "cuda"

    model = SemanticTokenLM(
        vocab_size=2048,
        num_codebooks=4,
        hidden_size=256,
        num_layers=4,
        num_heads=4,
        max_seq_len=1024,
        style_dim=512,
        cross_dim=256,
        text_dim=768
    ).to(device)

    start_epoch = 0
    global_step = 0
    best_loss = float("inf")

    frontend = AudioAnalysisFrontend(sample_rate=48000, device=device)
    text_encoder = TextConditioningEncoder(in_dim=768, out_dim=768).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    if resume:
        lm_ckpt = os.path.join(save_dir, "lm_latest.pt")
        if os.path.exists(lm_ckpt):
            state = load_model_checkpoint(model, lm_ckpt, optimizer=optimizer, device=device)
            if state and "epoch" in state:
                start_epoch = state["epoch"]
                global_step = state.get("step", 0)
                print(f"Resuming LM from Epoch {start_epoch}, Global Step {global_step}")

    if compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception:
            pass

    lr_scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    encoder = SEANetEncoder(in_channels=2, base_channels=64, latent_dim=64, ratios=[8, 5, 4, 3, 2]).to(device)
    codec_ckpt = os.path.join(save_dir, "codec_encoder_latest.pt")
    if os.path.exists(codec_ckpt):
        load_model_checkpoint(encoder, codec_ckpt, device=device)
    encoder.eval()

    quantizer = ResidualVectorQuantizer(num_codebooks=4, codebook_size=2048, dim=64).to(device)
    quantizer.eval()

    dataset = MusicAudioDataset(root_dir=data_dir, duration_seconds=4.0)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model.train()
    for epoch in range(start_epoch, epochs):
        epoch_loss = 0.0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(dataloader):
            audio = batch["audio"].to(device)
            texts = batch["text"]

            with torch.no_grad():
                text_embs = text_encoder(texts, device=device)
                analysis = frontend.analyze(audio[0])
                style = analysis["clap"].unsqueeze(0).repeat(audio.shape[0], 1)
                z, _, _ = encoder(audio, sample=False)
                _, tokens, _ = quantizer(z.permute(0, 2, 1))

            with torch.amp.autocast(device_type=device.type, enabled=use_cuda, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                logits, _ = model(tokens, style, text_emb=text_embs, condition_dropout=0.1)
                loss = 0.0
                for k in range(4):
                    loss = loss + F.cross_entropy(logits[:, :, k, :].reshape(-1, 2048), tokens[:, :, k].reshape(-1))
                loss = loss / grad_accum

            scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(dataloader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_loss += loss.item() * grad_accum
            global_step += 1

            if batch_idx % 10 == 0:
                print(f"LM Epoch {epoch + 1}/{epochs} | Step {global_step} | Cross-Entropy Loss: {loss.item() * grad_accum:.4f}")

        lr_scheduler.step()
        avg_loss = epoch_loss / max(len(dataloader), 1)
        save_model_checkpoint(model, os.path.join(save_dir, "lm_latest.pt"), optimizer, epoch + 1, global_step)

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model_checkpoint(model, os.path.join(save_dir, "lm_best.pt"), optimizer, epoch + 1, global_step)

    print("LM Full Training Finished Successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./gtzan")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train_lm(args.data_dir, args.epochs, args.batch_size, args.lr, args.save_dir, args.device, args.compile, args.grad_accum, args.resume)
