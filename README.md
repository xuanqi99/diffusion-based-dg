# Diffusion-Based Domain Generalization

This repository contains a method for domain generalization based on diffusion
generative models. The goal is to improve model robustness when the training
and target domains have different visual styles, distributions, or acquisition
conditions.

## Overview

Domain generalization aims to train a model that can perform well on unseen or
weakly observed target domains. This project explores a diffusion-based data
augmentation strategy: a diffusion generative model is used to synthesize
target-domain-like samples, and the generated data is added to the training
pipeline as augmentation data for the target domain.

By increasing the diversity of target-style samples, the downstream model can
learn features that are less dependent on a single source domain and more
transferable across domain shifts.

## Method

The main idea is:

1. Train or use a diffusion generative model capable of producing samples with
   target-domain characteristics.
2. Generate additional target-domain or target-domain-like data.
3. Combine the generated samples with the original training data.
4. Train the downstream recognition model with the augmented dataset.
5. Evaluate whether the generated data improves generalization under domain
   shift.

## Motivation

Diffusion models provide strong generative quality and can model complex visual
distributions. In a domain generalization setting, this makes them useful for
creating diverse augmentation samples that simulate target-domain variations,
such as changes in style, texture, lighting, background, or noise.

## Intended Use

This project is intended for research on diffusion-based data augmentation for
domain generalization, especially scenarios where synthetic target-domain data
can help improve model performance under distribution shift.
