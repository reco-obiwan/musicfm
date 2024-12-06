# Music Foundation Model

### Wav2Vec2ConformerConfig


```json
{
  "activation_dropout": 0.1,
  "adapter_kernel_size": 3,
  "adapter_stride": 2,
  "add_adapter": false,
  "apply_spec_augment": true,
  "architectures": ["Wav2Vec2ConformerForCTC"],
  "attention_dropout": 0.1,
  "bos_token_id": 1,
  "classifier_proj_size": 256,
  "codevector_dim": 768,
  "conformer_conv_dropout": 0.1,
  "contrastive_logits_temperature": 0.1,
  "conv_bias": true,
  "conv_depthwise_kernel_size": 31,
  "conv_dim": [512, 512, 512, 512, 512, 512, 512],
  "conv_kernel": [10, 3, 3, 3, 3, 2, 2],
  "conv_stride": [5, 2, 2, 2, 2, 2, 2],
  "ctc_loss_reduction": "sum",
  "ctc_zero_infinity": false,
  "diversity_loss_weight": 0.1,
  "do_stable_layer_norm": true,
  "eos_token_id": 2,
  "feat_extract_activation": "gelu",
  "feat_extract_dropout": 0.0,
  "feat_extract_norm": "layer",
  "feat_proj_dropout": 0.1,
  "feat_quantizer_dropout": 0.0,
  "final_dropout": 0.1,
  "gradient_checkpointing": false,
  "hidden_act": "swish",
  "hidden_dropout": 0.1,
  "hidden_dropout_prob": 0.1,
  "hidden_size": 1024,
  "initializer_range": 0.02,
  "intermediate_size": 4096,
  "layer_norm_eps": 1e-5,
  "layerdrop": 0.0,
  "mask_feature_length": 10,
  "mask_feature_min_masks": 0,
  "mask_feature_prob": 0.0,
  "mask_time_length": 10,
  "mask_time_min_masks": 2,
  "mask_time_prob": 0.05,
  "max_source_positions": 5000,
  "model_type": "wav2vec2-conformer",
  "num_adapter_layers": 3,
  "num_attention_heads": 16,
  "num_codevector_groups": 2,
  "num_codevectors_per_group": 320,
  "num_conv_pos_embedding_groups": 16,
  "num_conv_pos_embeddings": 128,
  "num_feat_extract_layers": 7,
  "num_hidden_layers": 24,
  "num_negatives": 100,
  "output_hidden_size": 1024,
  "pad_token_id": 0,
  "position_embeddings_type": "rotary",
  "proj_codevector_dim": 768,
  "rotary_embedding_base": 10000,
  "tdnn_dilation": [1, 2, 3, 1, 1],
  "tdnn_dim": [512, 512, 512, 512, 1500],
  "tdnn_kernel": [5, 3, 3, 1, 1],
  "torch_dtype": "float32",
  "transformers_version": "4.46.1",
  "use_weighted_layer_sum": false,
  "vocab_size": 32,
  "xvector_output_dim": 512
}
```