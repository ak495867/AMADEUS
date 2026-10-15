import argparse
import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from amadeus.models.codec.seanet import SEANetEncoder
from amadeus.models.codec.decoder import HiFiGANCodecDecoder
from amadeus.models.codec.losses import CodecLossSuite
from amadeus.data.dataset import MusicAudioDataset, DataLoader
from amadeus.pipeline.checkpoint import save_model_checkpoint, load_model_checkpoint

def train_codec(data_dir="./dataset", epochs=2, batch_size=2, lr=2e-4, save_dir="./checkpoints", device="cpu", compile_model=False, grad_accum=1, resume=False):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")
    use_cuda = device.type == "cuda"

    encoder = SEANetEncoder(in_channels=2, base_channels=64, latent_dim=64, ratios=[8, 5, 4, 3, 2]).to(device)
    decoder = HiFiGANCodecDecoder(latent_dim=64, base_channels=128, out_channels=2, ratios=[2, 3, 4, 5, 8], enable_stems=True).to(device)
    loss_suite = CodecLossSuite().to(device)

    start_epoch = 0
    global_step = 0
    best_loss = float("inf")

    opt_g = optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=lr, betas=(0.8, 0.99))
    opt_d = optim.AdamW(loss_suite.mpd.parameters(), lr=lr, betas=(0.8, 0.99))

    if resume:
        enc_ckpt = os.path.join(save_dir, "codec_encoder_latest.pt")
        dec_ckpt = os.path.join(save_dir, "codec_decoder_latest.pt")
        if os.path.exists(enc_ckpt) and os.path.exists(dec_ckpt):
            state = load_model_checkpoint(encoder, enc_ckpt, optimizer=opt_g, device=device)
            load_model_checkpoint(decoder, dec_ckpt, device=device)
            if state and "epoch" in state:
                start_epoch = state["epoch"]
                global_step = state.get("step", 0)
                print(f"Resuming Codec from Epoch {start_epoch}, Global Step {global_step}")

    if compile_model and hasattr(torch, "compile"):
        try:
            encoder = torch.compile(encoder)
            decoder = torch.compile(decoder)
        except Exception:
            pass

    scheduler_g = CosineAnnealingLR(opt_g, T_max=max(epochs, 1), eta_min=1e-6)
    scheduler_d = CosineAnnealingLR(opt_d, T_max=max(epochs, 1), eta_min=1e-6)

    scaler_g = torch.amp.GradScaler("cuda", enabled=use_cuda)
    scaler_d = torch.amp.GradScaler("cuda", enabled=use_cuda)

    dataset = MusicAudioDataset(root_dir=data_dir, duration_seconds=4.0)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    encoder.train()
    decoder.train()
    loss_suite.train()

    for epoch in range(start_epoch, epochs):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        opt_g.zero_grad()
        opt_d.zero_grad()

        for batch_idx, batch in enumerate(dataloader):
            x = batch["audio"].to(device)

            with torch.amp.autocast(device_type=device.type, enabled=use_cuda, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                z, mu, logvar = encoder(x)
                x_pred = decoder(z)
                loss_g, _ = loss_suite.generator_loss(x_pred, x, mu, logvar)
                loss_g = loss_g / grad_accum

            scaler_g.scale(loss_g).backward()

            if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(dataloader):
                scaler_g.unscale_(opt_g)
                torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), max_norm=1.0)
                scaler_g.step(opt_g)
                scaler_g.update()
                opt_g.zero_grad()

            with torch.amp.autocast(device_type=device.type, enabled=use_cuda, dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
                loss_d = loss_suite.discriminator_loss(x_pred.detach(), x)
                loss_d = loss_d / grad_accum

            scaler_d.scale(loss_d).backward()

            if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(dataloader):
                scaler_d.unscale_(opt_d)
                torch.nn.utils.clip_grad_norm_(loss_suite.mpd.parameters(), max_norm=1.0)
                scaler_d.step(opt_d)
                scaler_d.update()
                opt_d.zero_grad()

            epoch_g_loss += loss_g.item() * grad_accum
            epoch_d_loss += loss_d.item() * grad_accum
            global_step += 1

            if batch_idx % 10 == 0:
                print(f"Codec Epoch {epoch + 1}/{epochs} | Step {global_step} | Loss G: {loss_g.item() * grad_accum:.4f} | Loss D: {loss_d.item() * grad_accum:.4f}")

        scheduler_g.step()
        scheduler_d.step()

        avg_loss = epoch_g_loss / max(len(dataloader), 1)
        save_model_checkpoint(encoder, os.path.join(save_dir, "codec_encoder_latest.pt"), opt_g, epoch + 1, global_step)
        save_model_checkpoint(decoder, os.path.join(save_dir, "codec_decoder_latest.pt"), opt_g, epoch + 1, global_step)

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model_checkpoint(encoder, os.path.join(save_dir, "codec_encoder_best.pt"), opt_g, epoch + 1, global_step)
            save_model_checkpoint(decoder, os.path.join(save_dir, "codec_decoder_best.pt"), opt_g, epoch + 1, global_step)

    print("Codec Full Training Finished Successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./gtzan")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train_codec(args.data_dir, args.epochs, args.batch_size, args.lr, args.save_dir, args.device, args.compile, args.grad_accum, args.resume)
