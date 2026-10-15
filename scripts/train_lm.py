import argparse
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from amadeus.models.lm.transformer import SemanticTokenLM
from amadeus.models.lm.conditioning import TextConditioningEncoder
from amadeus.models.rvq.quantizer import ResidualVectorQuantizer
from amadeus.data.dataset import MusicAudioDataset, DataLoader
from amadeus.data.analysis.frontend import AudioAnalysisFrontend
from amadeus.pipeline.checkpoint import save_model_checkpoint

def train_lm(data_dir="./dataset", epochs=2, batch_size=2, lr=1.5e-4, save_dir="./checkpoints", device="cpu"):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")

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

    frontend = AudioAnalysisFrontend(sample_rate=48000, device=device)
    text_encoder = TextConditioningEncoder(in_dim=768, out_dim=768).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    dataset = MusicAudioDataset(root_dir=data_dir, duration_seconds=4.0)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model.train()
    global_step = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            audio = batch["audio"].to(device)
            texts = batch["text"]

            with torch.no_grad():
                text_embs = text_encoder(texts, device=device)
                analysis = frontend.analyze(audio[0])
                style = analysis["clap"].unsqueeze(0).repeat(audio.shape[0], 1)

            seq_len = 64
            tokens = torch.randint(0, 2048, (audio.shape[0], seq_len, 4), device=device)

            optimizer.zero_grad()
            logits = model(tokens, style, text_emb=text_embs, condition_dropout=0.1)

            loss = 0.0
            for k in range(4):
                loss = loss + F.cross_entropy(logits[:, :, k, :].reshape(-1, 2048), tokens[:, :, k].reshape(-1))

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            global_step += 1

            if batch_idx % 10 == 0:
                print(f"LM Epoch {epoch + 1}/{epochs} | Batch {batch_idx} | Cross-Entropy Loss: {loss.item():.4f}")

        save_model_checkpoint(model, os.path.join(save_dir, f"lm_epoch_{epoch+1}.pt"), optimizer, epoch+1, global_step)

    save_model_checkpoint(model, os.path.join(save_dir, "lm_latest.pt"), optimizer, epochs, global_step)
    print("LM Training Finished Successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./fma_dataset")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    train_lm(args.data_dir, args.epochs, args.batch_size, args.lr, args.save_dir, args.device)
