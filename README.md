# Music Foundation Model



## 학습 모니터링

https://wandb.ai/dreamus-company/musicfm_v01

## Reference

- https://github.com/minzwon/musicfm
- https://arxiv.org/abs/2202.01855
- https://arxiv.org/abs/2311.03318

## MusicFM 모델 설정 값

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



### Mespectogram statics for normalization

```json
{
    "spec_256_cnt": 14394344256,
    "spec_256_mean": -23.34296658431829,
    "spec_256_std": 26.189295587132637,
    "spec_512_cnt": 28677104448,
    "spec_512_mean": -21.31267396860235,
    "spec_512_std": 26.52644536245769,
    "spec_1024_cnt": 57242624832,
    "spec_1024_mean": -18.852271129208273,
    "spec_1024_std": 26.443154583585663,
    "spec_2048_cnt": 114373665600,
    "spec_2048_mean": -15.638743433896792,
    "spec_2048_std": 26.115825961611545,
    "spec_4096_cnt": 228635747136,
    "spec_4096_mean": -11.715532502794836,
    "spec_4096_std": 25.763972210234062,
    "melspec_256_cnt": 14282760192,
    "melspec_256_mean": -26.962600400166156,
    "melspec_256_std": 36.13614100912126,
    "melspec_512_cnt": 14282760192,
    "melspec_512_mean": -9.108344167718862,
    "melspec_512_std": 24.71910937988429,
    "melspec_1024_cnt": 14282760192,
    "melspec_1024_mean": 0.37302579246531126,
    "melspec_1024_std": 18.684082325919388,
    "melspec_2048_cnt": 14282760192,
    "melspec_2048_mean": 6.768444971712967,
    "melspec_2048_std": 18.417922652295623,
    "melspec_4096_cnt": 14282760192,
    "melspec_4096_mean": 13.617164614990036,
    "melspec_4096_std": 18.08552130124525,
    "cqt_cnt": 9373061376,
    "cqt_mean": 0.46341379757927165,
    "cqt_std": 0.9543998080910191,
    "mfcc_256_cnt": 1339008768,
    "mfcc_256_mean": -11.681755459447485,
    "mfcc_256_std": 29.183186444668316,
    "mfcc_512_cnt": 1339008768,
    "mfcc_512_mean": -2.540581461792183,
    "mfcc_512_std": 31.93752185832081,
    "mfcc_1024_cnt": 1339008768,
    "mfcc_1024_mean": 6.606636263169779,
    "mfcc_1024_std": 34.151644801729624,
    "mfcc_2048_cnt": 1339008768,
    "mfcc_2048_mean": 5.281600844245184,
    "mfcc_2048_std": 33.12784541220003,
    "mfcc_4096_cnt": 1339008768,
    "mfcc_4096_mean": 4.7616569480166095,
    "mfcc_4096_std": 32.61458906894133,
    "chromagram_256_cnt": 1339008768,
    "chromagram_256_mean": 55.15596556703181,
    "chromagram_256_std": 73.91858278719991,
    "chromagram_512_cnt": 1339008768,
    "chromagram_512_mean": 175.73092252759895,
    "chromagram_512_std": 248.48485148525953,
    "chromagram_1024_cnt": 1339008768,
    "chromagram_1024_mean": 589.2947481634608,
    "chromagram_1024_std": 913.857929063196,
    "chromagram_2048_cnt": 1339008768,
    "chromagram_2048_mean": 2062.286388327397,
    "chromagram_2048_std": 3458.92657915397,
    "chromagram_4096_cnt": 1339008768,
    "chromagram_4096_mean": 7673.039107997085,
    "chromagram_4096_std": 13009.883158267234
}

```

