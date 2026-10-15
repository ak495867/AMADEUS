import argparse
import os
import torch
import torch.optim as optim
from amadeus.models.codec.seanet import SEANetEncoder
from amadeus.models.codec.decoder import HiFiGANCodecDecoder
from amadeus.models.codec.losses import CodecLossSuite
from amadeus.data.dataset import MusicAudioDataset, DataLoader
from amadeus.pipeline.checkpoint import save_model_checkpoint

def train_codec(data_dir="./dataset", epochs=2, batch_size=2, lr=2e-4, save_dir="./checkpoints", device="cpu"):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")

    encoder = SEANetEncoder(in_channels=2, base_channels=64, latent_dim=64, ratios=[8, 5, 4, 3, 2]).to(device)
    decoder = HiFiGANCodecDecoder(latent_dim=64, base_channels=128, out_channels=2, ratios=[2, 3, 4, 5, 8], enable_stems=True).to(device)
    loss_suite = CodecLossSuite().to(device)

    opt_g = optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=lr, betas=(0.8, 0.99))
    opt_d = optim.AdamW(loss_suite.mpd.parameters(), lr=lr, betas=(0.8, 0.99))

    dataset = MusicAudioDataset(root_dir=data_dir, duration_seconds=4.0)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    encoder.train()
    decoder.train()
    loss_suite.train()

    global_step = 0
    for epoch in range(epochs):
        epoch_g_loss = 0.0
        epoch_d_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            x = batch["audio"].to(device)

            opt_g.zero_grad()
            z, mu, logvar = encoder(x)
            x_pred = decoder(z)
            loss_g, loss_dict = loss_suite.generator_loss(x_pred, x, mu, logvar)
            loss_g.backward()
            opt_g.step()

            opt_d.zero_grad()
            loss_d = loss_suite.discriminator_loss(x_pred.detach(), x)
            loss_d.backward()
            opt_d.step()

            epoch_g_loss += loss_g.item()
            epoch_d_loss += loss_d.item()
            global_step += 1

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} | Batch {batch_idx} | Loss G: {loss_g.item():.4f} | Loss D: {loss_d.item():.4f}")

        save_model_checkpoint(encoder, os.path.join(save_dir, f"codec_encoder_epoch_{epoch+1}.pt"), opt_g, epoch+1, global_step)
        save_model_checkpoint(decoder, os.path.join(save_dir, f"codec_decoder_epoch_{epoch+1}.pt"), opt_g, epoch+1, global_step)

    save_model_checkpoint(encoder, os.path.join(save_dir, "codec_encoder_latest.pt"), opt_g, epochs, global_step)
    save_model_checkpoint(decoder, os.path.join(save_dir, "codec_decoder_latest.pt"), opt_g, epochs, global_step)
    print("Codec Training Finished Successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./fma_dataset")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    train_codec(args.data_dir, args.epochs, args.batch_size, args.lr, args.save_dir, args.device)
