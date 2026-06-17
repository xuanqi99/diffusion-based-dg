"""Utilities for Cross Domain Generative Augmentation (CDGA).

CDGA is an offline data augmentation strategy for domain generalization. For
each source-domain image, a latent diffusion model generates synthetic images
guided toward other source domains of the same class. The original and generated
images are then used together with a standard ERM training loop.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
from pathlib import Path
import random
import shutil
from typing import Callable, Iterable, Mapping, Sequence


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}

DEFAULT_DOMAIN_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "PACS": {
        "A": "art painting",
        "C": "cartoon, cartoonish",
        "P": "photorealistic, extremely detailed",
        "S": "sketch drawing, black and white, less details",
    },
    "OfficeHome": {
        "Art": "art painting, art",
        "Clipart": "clipart, schematic, simplified",
        "Product": "product, merchandise",
        "Real": "real world, extremely detailed",
    },
    "DomainNet": {
        "clipart": "cartoon, cartoonish, drawing",
        "infograph": "infographic, data visualization, poster",
        "painting": "art painting",
        "quickdraw": "extremely simple drawing, black and white",
        "real": "photorealistic, extremely detailed",
        "sketch": "sketch drawing, black and white, less details",
    },
}


@dataclass(frozen=True)
class DomainExample:
    """One image from an ImageFolder-style domain dataset."""

    path: Path
    domain: str
    label: str


@dataclass(frozen=True)
class CDGAJob:
    """One synthetic image generation request."""

    source_path: Path
    output_path: Path
    label: str
    source_domain: str
    target_domain: str
    prompt: str | None = None
    guidance_path: Path | None = None
    seed: int | None = None

    @property
    def generated_domain(self) -> str:
        return f"{self.source_domain}_{self.target_domain}"


PromptGuidedGenerator = Callable[[Path, str, Path, CDGAJob], None]
ImageGuidedGenerator = Callable[[Path, Path, Path, CDGAJob], None]


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _safe_name(value: str) -> str:
    allowed = []
    for character in value.replace(" ", "_"):
        if character.isalnum() or character in {"-", "_"}:
            allowed.append(character)
    return "".join(allowed) or "item"


def _stable_digest(path: Path) -> str:
    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:10]


def _hashed_filename(path: Path) -> str:
    suffix = path.suffix.lower() or ".png"
    return f"{_safe_name(path.stem)}_{_stable_digest(path)}{suffix}"


def scan_imagefolder_domains(root: str | Path) -> list[DomainExample]:
    """Scan ``root/domain/class/image`` into a flat list of examples."""

    root = Path(root)
    examples: list[DomainExample] = []
    for domain_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for class_dir in sorted(path for path in domain_dir.iterdir() if path.is_dir()):
            for image_path in sorted(path for path in class_dir.rglob("*") if _is_image(path)):
                examples.append(
                    DomainExample(
                        path=image_path,
                        domain=domain_dir.name,
                        label=class_dir.name,
                    )
                )
    return examples


class CDGAPromptBuilder:
    """Build prompt-guided CDGA prompts from class and domain descriptions."""

    def __init__(
        self,
        domain_descriptions: Mapping[str, str],
        *,
        template: str = "a {label}, {domain_description}",
    ) -> None:
        self.domain_descriptions = dict(domain_descriptions)
        self.template = template

    @classmethod
    def from_dataset(cls, dataset_name: str) -> "CDGAPromptBuilder":
        if dataset_name not in DEFAULT_DOMAIN_DESCRIPTIONS:
            known = ", ".join(sorted(DEFAULT_DOMAIN_DESCRIPTIONS))
            raise KeyError(f"Unknown dataset {dataset_name!r}. Known datasets: {known}")
        return cls(DEFAULT_DOMAIN_DESCRIPTIONS[dataset_name])

    def __call__(self, label: str, target_domain: str) -> str:
        description = self.domain_descriptions.get(target_domain, target_domain)
        readable_label = label.replace("_", " ").replace("-", " ")
        return self.template.format(
            label=readable_label,
            target_domain=target_domain,
            domain_description=description,
        )


class CDGAPlanner:
    """Create and run offline CDGA generation jobs.

    The output layout is compatible with the paper's DomainBed-style loader:
    ``output_root/source_domain/source_target/class/image.png``.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        output_root: str | Path,
        *,
        prompt_builder: CDGAPromptBuilder | None = None,
        generation_batch_size: int = 1,
        include_self_domain: bool = True,
        seed: int = 0,
    ) -> None:
        if generation_batch_size < 1:
            raise ValueError("generation_batch_size must be at least 1.")

        self.dataset_root = Path(dataset_root)
        self.output_root = Path(output_root)
        self.prompt_builder = prompt_builder
        self.generation_batch_size = generation_batch_size
        self.include_self_domain = include_self_domain
        self.seed = seed
        self.examples = scan_imagefolder_domains(self.dataset_root)
        if not self.examples:
            raise ValueError(f"No images found under {self.dataset_root}.")
        self.domains = sorted({example.domain for example in self.examples})
        self.labels = sorted({example.label for example in self.examples})
        self._by_domain_label = self._index_examples(self.examples)

    @staticmethod
    def _index_examples(
        examples: Sequence[DomainExample],
    ) -> dict[tuple[str, str], list[DomainExample]]:
        index: dict[tuple[str, str], list[DomainExample]] = defaultdict(list)
        for example in examples:
            index[(example.domain, example.label)].append(example)
        return index

    def _resolve_domains(self, domains: Iterable[str] | None) -> list[str]:
        resolved = list(domains) if domains is not None else list(self.domains)
        unknown = sorted(set(resolved) - set(self.domains))
        if unknown:
            raise ValueError(f"Unknown domain(s): {unknown}. Known domains: {self.domains}")
        return resolved

    def _output_path(self, example: DomainExample, target_domain: str, replica: int) -> Path:
        filename = (
            f"{_safe_name(example.path.stem)}_"
            f"{_stable_digest(example.path)}_"
            f"{replica:03d}{example.path.suffix.lower() or '.png'}"
        )
        return (
            self.output_root
            / example.domain
            / f"{example.domain}_{target_domain}"
            / example.label
            / filename
        )

    def copy_original_domains(self, domains: Iterable[str] | None = None) -> None:
        """Copy original images into ``domain/domain/class`` folders."""

        selected_domains = set(self._resolve_domains(domains))
        for example in self.examples:
            if example.domain not in selected_domains:
                continue
            destination = (
                self.output_root
                / example.domain
                / example.domain
                / example.label
                / _hashed_filename(example.path)
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(example.path, destination)

    def prompt_guided_jobs(
        self,
        *,
        source_domains: Iterable[str] | None = None,
        target_domains: Iterable[str] | None = None,
    ) -> list[CDGAJob]:
        """Build CDGA-PG jobs for all source/target domain pairs."""

        if self.prompt_builder is None:
            raise ValueError("prompt_builder is required for prompt-guided CDGA.")

        sources = set(self._resolve_domains(source_domains))
        targets = self._resolve_domains(target_domains)
        jobs: list[CDGAJob] = []
        for example in self.examples:
            if example.domain not in sources:
                continue
            for target_domain in targets:
                if not self.include_self_domain and target_domain == example.domain:
                    continue
                for replica in range(self.generation_batch_size):
                    jobs.append(
                        CDGAJob(
                            source_path=example.path,
                            output_path=self._output_path(example, target_domain, replica),
                            label=example.label,
                            source_domain=example.domain,
                            target_domain=target_domain,
                            prompt=self.prompt_builder(example.label, target_domain),
                            seed=self.seed + len(jobs),
                        )
                    )
        return jobs

    def image_guided_jobs(
        self,
        *,
        source_domains: Iterable[str] | None = None,
        target_domains: Iterable[str] | None = None,
    ) -> list[CDGAJob]:
        """Build CDGA-IG jobs using same-class images from target domains."""

        sources = set(self._resolve_domains(source_domains))
        targets = self._resolve_domains(target_domains)
        rng = random.Random(self.seed)
        jobs: list[CDGAJob] = []
        for example in self.examples:
            if example.domain not in sources:
                continue
            for target_domain in targets:
                if not self.include_self_domain and target_domain == example.domain:
                    continue
                guidance_pool = self._by_domain_label.get((target_domain, example.label), [])
                if not guidance_pool:
                    continue
                for replica in range(self.generation_batch_size):
                    guidance = rng.choice(guidance_pool)
                    jobs.append(
                        CDGAJob(
                            source_path=example.path,
                            output_path=self._output_path(example, target_domain, replica),
                            label=example.label,
                            source_domain=example.domain,
                            target_domain=target_domain,
                            guidance_path=guidance.path,
                            seed=self.seed + len(jobs),
                        )
                    )
        return jobs

    @staticmethod
    def run_prompt_guided(
        jobs: Iterable[CDGAJob],
        generator: PromptGuidedGenerator,
        *,
        skip_existing: bool = True,
    ) -> int:
        """Run CDGA-PG jobs with a user-provided image-to-image generator."""

        completed = 0
        for job in jobs:
            if job.prompt is None:
                raise ValueError(f"Job for {job.source_path} has no prompt.")
            if skip_existing and job.output_path.exists():
                continue
            job.output_path.parent.mkdir(parents=True, exist_ok=True)
            generator(job.source_path, job.prompt, job.output_path, job)
            completed += 1
        return completed

    @staticmethod
    def run_image_guided(
        jobs: Iterable[CDGAJob],
        generator: ImageGuidedGenerator,
        *,
        skip_existing: bool = True,
    ) -> int:
        """Run CDGA-IG jobs with a user-provided image-mixer generator."""

        completed = 0
        for job in jobs:
            if job.guidance_path is None:
                raise ValueError(f"Job for {job.source_path} has no guidance image.")
            if skip_existing and job.output_path.exists():
                continue
            job.output_path.parent.mkdir(parents=True, exist_ok=True)
            generator(job.source_path, job.guidance_path, job.output_path, job)
            completed += 1
        return completed


def build_cdga_imagefolder_datasets(
    cdga_root: str | Path,
    test_domains: Iterable[str | int],
    *,
    augment: bool = False,
    transform: object | None = None,
    augment_transform: object | None = None,
    include_target_guidance: bool = False,
) -> list[object]:
    """Build DomainBed-style datasets from a generated CDGA folder.

    Training domains are loaded from all generated subdomains except those whose
    folder name contains the held-out test domain, matching standard CDGA. Set
    ``include_target_guidance=True`` for CDGA-star experiments where target
    domain descriptions were intentionally used during generation.
    """

    from torch.utils.data import ConcatDataset
    from torchvision import transforms
    from torchvision.datasets import ImageFolder

    cdga_root = Path(cdga_root)
    environments = sorted(path.name for path in cdga_root.iterdir() if path.is_dir())
    test_domain_names = _normalize_test_domains(test_domains, environments)

    if transform is None:
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
    if augment_transform is None:
        augment_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
                transforms.RandomGrayscale(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    datasets: list[object] = []
    for environment in environments:
        env_transform = augment_transform if augment and environment not in test_domain_names else transform
        env_path = cdga_root / environment

        if environment in test_domain_names:
            datasets.append(ImageFolder(env_path / environment, transform=env_transform))
            continue

        subdatasets = []
        for subdomain_path in sorted(path for path in env_path.iterdir() if path.is_dir()):
            uses_test_guidance = _subdomain_uses_guidance(subdomain_path.name, test_domain_names)
            if include_target_guidance or not uses_test_guidance:
                subdatasets.append(ImageFolder(subdomain_path, transform=env_transform))
        if not subdatasets:
            raise ValueError(f"No CDGA subdomains selected for training domain {environment!r}.")
        datasets.append(ConcatDataset(subdatasets))

    return datasets


def _normalize_test_domains(test_domains: Iterable[str | int], environments: Sequence[str]) -> set[str]:
    names: set[str] = set()
    for domain in test_domains:
        if isinstance(domain, int):
            names.add(environments[domain])
        else:
            names.add(domain)
    unknown = sorted(names - set(environments))
    if unknown:
        raise ValueError(f"Unknown test domain(s): {unknown}. Known domains: {environments}")
    return names


def _subdomain_uses_guidance(subdomain: str, test_domain_names: set[str]) -> bool:
    if subdomain in test_domain_names:
        return False
    if "_" in subdomain:
        _source, target = subdomain.split("_", 1)
        return target in test_domain_names
    return False


def make_diffusers_img2img_generator(
    pipeline: object,
    *,
    strength: float = 0.75,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
) -> PromptGuidedGenerator:
    """Adapt a diffusers image-to-image pipeline to ``run_prompt_guided``."""

    from PIL import Image
    import torch

    def generate(source_path: Path, prompt: str, output_path: Path, job: CDGAJob) -> None:
        image = Image.open(source_path).convert("RGB")
        torch_generator = None
        if job.seed is not None:
            torch_generator = torch.Generator(device=getattr(pipeline, "device", "cpu"))
            torch_generator.manual_seed(job.seed)

        result = pipeline(
            prompt=prompt,
            image=image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=torch_generator,
        )
        result.images[0].save(output_path)

    return generate
