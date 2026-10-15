import argparse
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from amadeus.models.dit.dit import DiffusionTransformer
from amadeus.models.dit.flow_matching import FlowMatchingScheduler
from amadeus.models.codec.seanet import SEANetEncoder
from amadeus.data.dataset import MusicAudioDataset, DataLoader
from amadeus.data.analysis.frontend import AudioAnalysisFrontend
from amadeus.pipeline.checkpoint import save_model_checkpoint

def train_dit(data_dir="./dataset", epochs=2, batch_size=2, lr=1e-4, save_dir="./checkpoints", device="cpu"):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")

    model = DiffusionTransformer(
        in_channels=64,
        hidden_size=256,
        num_layers=4,
        num_heads=4,
        patch_size=4,
        style_dim=512,
        cross_dim=256
    ).to(device)

    encoder = SEANetEncoder(in_channels=2, base_channels=64, latent_dim=64, ratios=[8, 5, 4, 3, 2]).to(device)
    encoder.eval()

    frontend = AudioAnalysisFrontend(sample_rate=48000, device=device)
    scheduler = FlowMatchingScheduler()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    dataset = MusicAudioDataset(root_dir=data_dir, duration_seconds=4.0)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model.train()
    global_step = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            audio = batch["audio"].to(device)
            with torch.no_grad():
                z, _, _ = encoder(audio, sample=False)
                analysis = frontend.analyze(audio[0])
                style = analysis["clap"].unsqueeze(0).repeat(audio.shape[0], 1)
                loudness = torch.tensor([[analysis["lufs"]]], device=device).repeat(audio.shape[0], 1)

            noise = torch.randn_like(z)
            t = scheduler.sample_timesteps(audio.shape[0], device=device)
            x_t, target_v = scheduler.add_noise(z, noise, t)

            optimizer.zero_grad()
            pred_v = model(x_t, t, style=style, loudness=loudness)
            loss = F.mse_loss(pred_v, target_v)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            if batch_idx % 10 == 0:
                print(f"DiT Epoch {epoch + 1}/{epochs} | Batch {batch_idx} | Flow Loss: {loss.item():.4f}")

        save_model_checkpoint(model, os.path.join(save_dir, f"dit_epoch_{epoch+1}.pt"), optimizer, epoch+1, global_step)

    save_model_checkpoint(model, os.path.join(save_dir, "dit_latest.pt"), optimizer, epochs, global_step)
    print("DiT Training Finished Successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./fma_dataset")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    train_dit(args.data_dir, args.epochs, args.batch_size, args.lr, args.save_dir, args.device)
