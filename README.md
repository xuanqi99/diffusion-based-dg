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
