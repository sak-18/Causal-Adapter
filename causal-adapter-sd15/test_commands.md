# Tested training commands (local)

This file is **gitignored** — it holds the real absolute paths I use on this
machine so the commands below can be copy-pasted into a terminal as-is.

Conventions:

- `--pretrained_model_name_or_path` is a **local** miniSD snapshot. Replace
  with a HuggingFace id (e.g. `runwayml/stable-diffusion-v1-5`) on a machine
  that can reach the Hub.
- `--scm_path` points at the SCM head produced by `SCM_modeling`.
- Anything that matches the default in `train.py` is omitted (run
  `python train.py --help` to see all flags).

Removed from the legacy commands (no longer accepted by argparse):
`--num_processes`, `--num_machines`, the duplicated `train.py` token, and
`--num_validation_images`.

## Pendulum

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --output_name "causal-adapter" \
  --output_dir "./logs/logs_pendulum_all" \
  --pretrained_model_name_or_path "/projects/dsai/se_aieng/cai/causal/workspace/causaledit/MCPL-diffuser/.cache/huggingface/hub/models--lambdalabs--miniSD-diffusers/snapshots/26ed8a9bfbf76f46a6cf60517dde321f900c44ce" \
  --train_data_dir "/projects/dsai/se_aieng/cai/causal/workspace/causaledit/MCPL-diffuser/dataset/causal_data2/pendulum/train/" \
  --dataset "pendulum" \
  --resolution 256 \
  --train_batch_size 2 \
  --max_train_steps 20000 \
  --placeholder_string 'a image of @ and * and & and !' \
  --presudo_words '@,*,&,!' \
  --presudo_words_infonce '@,*,&,!' \
  --scm_path "/projects/dsai/se_aieng/cai/causal/workspace/causaledit/MCPL-diffuser/logs/logs_pendulum_all/2025-06-02T11-36-21-causalnet_pretrain_floatGaussian/best_model.pt"
```

## ADNI

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --output_name "controlnet_textcond_contrast_nohorizonflip" \
  --output_dir "./logs/logs_ADNI_all" \
  --pretrained_model_name_or_path "/projects/dsai/se_aieng/cai/causal/workspace/causaledit/MCPL-diffuser/.cache/huggingface/hub/models--lambdalabs--miniSD-diffusers/snapshots/26ed8a9bfbf76f46a6cf60517dde321f900c44ce" \
  --train_data_dir "/projects/dsai/se_aieng/cai/causal/workspace/causaledit/counterfactual-benchmark/counterfactual_benchmark/ctf_datasets/adni/preprocessing" \
  --dataset "ADNI" \
  --resolution 256 \
  --train_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --max_train_steps 100000 \
  --placeholder_string 'a mri image of @ and * and &' \
  --presudo_words '@,*,&' \
  --presudo_words_infonce '@,*,&' \
  --scm_path "/projects/dsai/se_aieng/cai/causal/workspace/causaledit/MCPL-diffuser/logs/logs_ADNI_all/2025-05-01T09-58-53-causalnet_pretrain/best_model.pt"
```

## CelebA (complex)

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --output_name "causal-adapter" \
  --output_dir "./logs/logs_celeA_complex_all" \
  --pretrained_model_name_or_path "/projects/dsai/se_aieng/cai/causal/workspace/causaledit/MCPL-diffuser/.cache/huggingface/hub/models--lambdalabs--miniSD-diffusers/snapshots/26ed8a9bfbf76f46a6cf60517dde321f900c44ce" \
  --train_data_dir "/projects/dsai/se_aieng/cai/causal/workspace/causaledit/counterfactual-benchmark/datasets/" \
  --dataset "celeA_complex" \
  --resolution 256 \
  --train_batch_size 2 \
  --max_train_steps 200000 \
  --learning_rate 5e-6 \
  --placeholder_string 'an image of @ * & !' \
  --presudo_words '@,*,&,!' \
  --presudo_words_infonce '@,*,&,!' \
  --scm_path "/projects/dsai/se_aieng/cai/causal/workspace/causaledit/MCPL-diffuser/logs/logs_celeA_complex_all/2025-04-23T21-11-19-causalnet_pretrain/best_model.pt"
```

## Notes

- All three commands run with `--mcpl_training True` (default), frozen SCM
  (`--causal_training False`, default), and
  `--task_cond generation_text_global_after` (default). Override only if
  you want different behaviour.
- The Pendulum command was smoke-tested for 10 steps on a Tesla V100-32GB —
  loss decreasing, NLL active, embeddings + controlnet saved.
- For a smoke test, append `--max_train_steps 10 --save_steps 999999` to
  any command above.
