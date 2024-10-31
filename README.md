# Cool-BERT

This project implements a Transformer-based, 
encoder-only model with Mixture of Experts (MoE) integration,
Rotary Positional Embeddings for Positional Encoding, and RMSnorm as 
the normalization layer.

- The model is currently trained on the Colossal Clean Crawled Corpus (C4) dataset using a streaming data loader with the Masked Language Modeling (MLM) objective.
- The training script is configured for a single GPU setup.
- Mixed precision training is implemented using PyTorch AMP.
- MLflow is used for tracking and logging.
- Expert imbalance is managed using the auxiliary loss from ["Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity"](https://arxiv.org/abs/2101.03961) and by injecting noise into the expert selection process.

## Future Improvements
- Update the training script to support data parallelism or pipeline parallelism.
- Fine-tune the model on classic downstream tasks.
- Adapt the model for embedding tasks.


