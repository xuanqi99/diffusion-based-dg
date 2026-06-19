# Diffusion-Based Domain Generalization

This repository contains a method for domain generalization based on diffusion
generative models. The goal is to improve model robustness on unseen target
domains without assuming access to target-domain data during training.

## Overview

Domain generalization aims to train a model that can perform well on unseen or
unknown target domains. In this setting, the target domain is not available
during training, so the method should not rely on knowing target-domain
characteristics in advance.

This project explores a diffusion-based data augmentation strategy for domain
generalization. Instead of generating samples for a known target domain, the
method uses source-domain data to synthesize diverse samples that represent
possible unknown domain variations. These generated unknown-domain samples are
added to the training pipeline as augmented data.

By expanding the diversity of the source-domain training distribution, the
downstream model can learn features that are less dependent on the observed
source domains and more transferable to unseen target domains.

## Method

The main idea is:

1. Use source-domain data as the available training data.
2. Train or use a diffusion generative model to explore plausible domain
   variations beyond the observed source domains.
3. Generate additional unknown-domain or domain-shifted samples.
4. Combine the generated samples with the original source-domain data.
5. Train the downstream recognition model with the augmented dataset.
6. Evaluate whether the generated data improves performance on unseen target
   domains.

## ERM DG Method

This repository includes a PyTorch implementation of Empirical Risk
Minimization (ERM), the standard baseline for domain generalization. ERM pools
labeled minibatches from all available source domains and minimizes the regular
supervised classification loss without using target-domain data.

The implementation is provided in [`erm.py`](erm.py). It supports both a single
training batch and a sequence of source-domain minibatches.

Minimal multi-domain usage:

```python
import torch

from erm import ERMTrainer, erm_step

model = build_model(num_classes=num_classes)
trainer = ERMTrainer(model)
optimizer = torch.optim.Adam(trainer.parameters(), lr=3e-5)

for source_minibatches in train_iterator:
    output = erm_step(
        trainer,
        optimizer,
        minibatches=source_minibatches,
    )
    print(output.loss.item(), output.accuracy.item())
```

Each item in `source_minibatches` must be an `(inputs, targets)` tuple. ERM
concatenates the domain batches before computing cross entropy, matching the
usual pooled-source DG baseline.

## CDGA DG Method

This repository includes a Python implementation of CDGA (Cross Domain
Generative Augmentation), a diffusion-based domain generalization method that
generates synthetic samples in the vicinity of source-domain pairs. CDGA first
creates cross-domain images with a latent diffusion model and then trains the
recognition model with standard ERM on both original and generated data.

The implementation is provided in [`cdga.py`](cdga.py). It supports:

- scanning ImageFolder-style datasets organized as `domain/class/image`;
- prompt-guided CDGA generation jobs, where the target domain is represented by
  a text prompt;
- image-guided CDGA generation jobs, where a same-class image from another
  domain is used as guidance;
- DomainBed-style dataset construction from generated CDGA folders.

Minimal prompt-guided usage:

```python
from diffusers import StableDiffusionImg2ImgPipeline

from cdga import CDGAPromptBuilder, CDGAPlanner, make_diffusers_img2img_generator

pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe = pipe.to("cuda")

planner = CDGAPlanner(
    dataset_root="data/PACS",
    output_root="data/PACS_CDGA",
    prompt_builder=CDGAPromptBuilder.from_dataset("PACS"),
    generation_batch_size=1,
    include_self_domain=False,
)

jobs = planner.prompt_guided_jobs()
generator = make_diffusers_img2img_generator(pipe, strength=0.75)
planner.copy_original_domains()
planner.run_prompt_guided(jobs, generator)
```

After generation, the resulting folder can be loaded for ERM training:

```python
from cdga import build_cdga_imagefolder_datasets

datasets = build_cdga_imagefolder_datasets(
    cdga_root="data/PACS_CDGA",
    test_domains=["S"],
    augment=True,
)
```

## MIRO DG Method

This repository also includes a PyTorch implementation of MIRO (Mutual
Information Regularization with Oracle), a domain generalization method that
uses a frozen pre-trained model as an oracle approximation. During supervised
training on source domains, MIRO adds a feature-level regularization term that
keeps the trainable model close to the oracle representation and reduces
overfitting to source-domain-specific cues.

The implementation is provided in [`miro.py`](miro.py). It is framework-light
and can wrap any `torch.nn.Module` whose intermediate layers can be addressed by
`named_modules()`.

Minimal usage:

```python
import copy
import torch

from miro import MIROTrainer, miro_step

student = build_model(num_classes=num_classes)
oracle = copy.deepcopy(student)  # usually the original pre-trained checkpoint

trainer = MIROTrainer(
    student=student,
    oracle=oracle,
    feature_layers=("layer1", "layer2", "layer3", "layer4"),
    lambda_miro=0.01,
)

optimizer = torch.optim.Adam(trainer.parameters(), lr=3e-5, weight_decay=0.0)

for images, labels in source_loader:
    output = miro_step(trainer, optimizer, images, labels)
    print(output.total_loss.item(), output.task_loss.item(), output.miro_loss.item())
```

For ResNet-style models, common feature layers are `layer1` through `layer4`.
For custom backbones, use `named_modules_containing(model, tokens)` from
`miro.py` to inspect candidate layer names.

## Motivation

Diffusion models provide strong generative quality and can model complex visual
distributions. In a domain generalization setting, this makes them useful for
creating diverse augmentation samples that explore possible domain shifts, such
as changes in style, texture, lighting, background, acquisition condition, or
noise.

The key assumption is not that the target domain is known, but that generated
domain diversity can help the model become more invariant to distribution
changes and therefore perform better on unknown target domains.

## Intended Use

This project is intended for research on diffusion-based data augmentation for
domain generalization, especially scenarios where synthetic unknown-domain data
can help improve model performance under distribution shift.

## References

- Hemati, S., Beitollahi, M., Estiri, A. H., Al Omari, B., Chen, X., and
  Zhang, G. "Cross Domain Generative Augmentation: Domain Generalization with
  Latent Diffusion Models." arXiv:2312.05387, 2023.
- Cha, J., Lee, K., Park, S., and Chun, S. "Domain Generalization by
  Mutual-Information Regularization with Pre-trained Models." ECCV 2022.
- Official MIRO implementation: https://github.com/khanrc/miro
