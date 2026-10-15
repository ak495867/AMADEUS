# AMADEUS Dataset Datasheet

## Motivation
- Purpose: Training high-fidelity, controllable musical generation models with strict provenance and licensing compliance.
- Corpus Size: 320,000 total hours across commercial catalogs, CC-BY / public domain, production libraries, and synthetic stems.

## Composition
- Licensed Commercial Catalog: 120,000 hours
- Creative Commons / Public Domain: 60,000 hours
- Production Music Libraries: 40,000 hours
- Synthetic Augmentation: 30,000 hours
- Paired Text & Audio Captions: 50,000 hours
- Instrument-labeled Timbre: 20,000 hours

## Ingestion and Rights Management
- Opt-out registry checked before ingestion.
- Deduplication threshold: 0.95 cosine similarity on CLAP embeddings.
- ITU-R BS.1770 loudness normalization to standard LUFS.
- WebDataset sharding format.
