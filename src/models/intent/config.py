"""
Configuration schemas for intent classification training and transfer-learning bootstrap.
"""

from pathlib import Path
from typing import Optional, Union
import yaml
from pydantic import BaseModel, ConfigDict, Field


class TrainConfig(BaseModel):
    """Configuration schema for intent classification transfer-learning bootstrap (M2.P2.1.F1).

    DistilBERT-base-uncased (66M parameters) is chosen as default over BERT-base (110M)
    because it runs ~60% faster on CPU and requires ~40% less memory while retaining
    over 97% of language understanding performance. The max_length=64 covers >99% of
    customer service inquiries without unnecessary padding overhead.
    """

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    # Model parameters
    model_name: str = Field(
        default="distilbert-base-uncased",
        description="Pretrained transformer model identifier (e.g. distilbert-base-uncased, bert-base-uncased)",
    )
    num_labels: int = Field(
        default=77,
        ge=2,
        description="Number of classification categories (77 for Banking77)",
    )
    max_length: int = Field(
        default=64,
        ge=8,
        le=512,
        description="Maximum sequence length for tokenization",
    )

    # Optimization & Training
    batch_size: int = Field(
        default=16,
        gt=0,
        description="Training batch size per device",
    )
    eval_batch_size: int = Field(
        default=32,
        gt=0,
        description="Evaluation batch size per device",
    )
    learning_rate: float = Field(
        default=2e-5,
        gt=0.0,
        description="Initial learning rate for AdamW optimizer",
    )
    weight_decay: float = Field(
        default=0.01,
        ge=0.0,
        description="Weight decay for AdamW optimizer",
    )
    epochs: int = Field(
        default=3,
        gt=0,
        description="Number of training epochs",
    )
    warmup_ratio: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Warmup ratio for learning rate scheduler",
    )
    patience: int = Field(
        default=2,
        ge=1,
        description="Early stopping patience based on validation macro-F1",
    )

    # Dataset & Split
    dataset_name: str = Field(
        default="PolyAI/banking77",
        description="HuggingFace dataset repository name",
    )
    local_dataset_path: Optional[str] = Field(
        default=None,
        description="Optional local filesystem path to Banking77 dataset for offline environments",
    )
    val_ratio: float = Field(
        default=0.15,
        gt=0.0,
        lt=0.5,
        description="Proportion of dataset reserved for validation split",
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility",
    )
    class_weighting: bool = Field(
        default=False,
        description="Whether to apply inverse frequency class weighting in loss calculation",
    )

    # Checkpointing & Metrics
    checkpoint_dir: str = Field(
        default="data/models/banking77_encoder",
        description="Target directory for saving the fine-tuned encoder checkpoint",
    )
    metrics_path: str = Field(
        default="data/models/banking77_metrics.json",
        description="Target path for exporting evaluation metrics and error analysis",
    )
    overwrite: bool = Field(
        default=False,
        description="Whether to overwrite existing checkpoint artifacts in checkpoint_dir",
    )

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "TrainConfig":
        """Load and validate training configuration from a YAML file.

        Parameters
        ----------
        yaml_path : Union[str, Path]
            Path to YAML configuration file.

        Returns
        -------
        TrainConfig
            Validated configuration instance.

        Raises
        ------
        FileNotFoundError
            If YAML file does not exist.
        ValueError
            If YAML content is invalid or missing required configuration.
        """
        path = Path(yaml_path)
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file not found at: {path.resolve()}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"YAML configuration at {path} must define a mapping/dictionary, got {type(data)}")

        return cls.model_validate(data)


class TrainingConfig(BaseModel):
    """Nested training settings for domain-adaptation fine-tuning."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    output_dir: str = Field(default="data/models/domain_adapted_encoder", description="Target directory for adaptation artifacts")
    learning_rate: float = Field(default=2e-5, gt=0.0)
    batch_size: int = Field(default=16, gt=0)
    eval_batch_size: int = Field(default=32, gt=0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    epochs: int = Field(default=3, gt=0)
    num_epochs: int | None = Field(default=None, gt=0)
    warmup_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    patience: int = Field(default=2, ge=1)
    val_ratio: float = Field(default=0.15, gt=0.0, lt=0.5)
    test_ratio: float = Field(default=0.15, gt=0.0, lt=0.5)
    seed: int = Field(default=42)
    overwrite: bool = Field(default=False)
    max_length: int = Field(default=64, ge=8, le=512, description="Maximum sequence length for tokenization")
    freeze_backbone: bool = Field(default=False, description="Whether to freeze backbone parameters during fine-tuning")


class TaxonomyConfig(BaseModel):
    """Configuration schema for taxonomy induction and domain-adapted fine-tuning (M2.P2.2.F1)."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    training: TrainingConfig = Field(default_factory=TrainingConfig)

    # Embedding & Clustering (HDBSCAN)
    min_cluster_size: int = Field(
        default=20,
        ge=2,
        description="Minimum size of clusters in HDBSCAN",
    )
    min_samples: Optional[int] = Field(
        default=5,
        ge=1,
        description="Number of samples in a neighborhood for a point to be considered a core point",
    )
    metric: str = Field(
        default="euclidean",
        description="Distance metric for HDBSCAN clustering",
    )
    cluster_selection_method: str = Field(
        default="eom",
        description="Cluster selection method for HDBSCAN ('eom' or 'leaf')",
    )

    # Taxonomy Induction & Labeling
    seed_labels: list[str] = Field(
        default=[
            "Device_Performance_Degradation",
            "Software_Update_Failure",
            "Account_Recovery",
            "Outrage_Escalation",
        ],
        description="Seed labels derived from PRD for taxonomy induction",
    )
    max_representative_texts: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum representative centroid-nearest samples passed to LLM for labeling",
    )
    purity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum purity score for cluster retention",
    )
    min_surviving_classes: int = Field(
        default=3,
        ge=2,
        description="Minimum number of classes required after filtering before raising a warning",
    )
    other_label: str = Field(
        default="Other_Uncategorized",
        description="Label used when merging sparse clusters",
    )
    unclassified_label: str = Field(
        default="Unclassified",
        description="Label assigned to noise/unclustered points (-1)",
    )

    # Severity Rules
    severity_keywords: list[str] = Field(
        default=[
            "lawyer",
            "attorney",
            "sue",
            "lawsuit",
            "legal action",
            "fire hazard",
            "caught fire",
            "battery explosion",
            "exploded",
            "injured",
            "burn",
            "hospital",
        ],
        description="Keywords indicating potential legal threat or physical/safety hazard",
    )

    # Domain Adaptation & Checkpoints
    encoder_checkpoint_dir: str = Field(
        default="data/models/banking77_encoder",
        description="Path to pretrained/bootstrapped encoder checkpoint",
    )
    domain_checkpoint_dir: str = Field(
        default="data/models/domain_adapted_encoder",
        description="Target directory for saving domain-adapted checkpoint",
    )
    taxonomy_path: str = Field(
        default="data/models/apple_taxonomy.json",
        description="Path to persist induced taxonomy JSON",
    )
    metrics_path: str = Field(
        default="data/models/domain_adapted_metrics.json",
        description="Path to persist domain adaptation metrics",
    )

    @property
    def output_dir(self) -> str:
        return self.training.output_dir

    @output_dir.setter
    def output_dir(self, value: str) -> None:
        self.training.output_dir = value

    @property
    def learning_rate(self) -> float:
        return self.training.learning_rate

    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        self.training.learning_rate = value

    @property
    def batch_size(self) -> int:
        return self.training.batch_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        self.training.batch_size = value

    @property
    def eval_batch_size(self) -> int:
        return self.training.eval_batch_size

    @eval_batch_size.setter
    def eval_batch_size(self, value: int) -> None:
        self.training.eval_batch_size = value

    @property
    def weight_decay(self) -> float:
        return self.training.weight_decay

    @weight_decay.setter
    def weight_decay(self, value: float) -> None:
        self.training.weight_decay = value

    @property
    def epochs(self) -> int:
        return self.training.epochs

    @epochs.setter
    def epochs(self, value: int) -> None:
        self.training.epochs = value

    @property
    def warmup_ratio(self) -> float:
        return self.training.warmup_ratio

    @warmup_ratio.setter
    def warmup_ratio(self, value: float) -> None:
        self.training.warmup_ratio = value

    @property
    def patience(self) -> int:
        return self.training.patience

    @patience.setter
    def patience(self, value: int) -> None:
        self.training.patience = value

    @property
    def val_ratio(self) -> float:
        return self.training.val_ratio

    @val_ratio.setter
    def val_ratio(self, value: float) -> None:
        self.training.val_ratio = value

    @property
    def test_ratio(self) -> float:
        return self.training.test_ratio

    @test_ratio.setter
    def test_ratio(self, value: float) -> None:
        self.training.test_ratio = value

    @property
    def seed(self) -> int:
        return self.training.seed

    @seed.setter
    def seed(self, value: int) -> None:
        self.training.seed = value

    @property
    def overwrite(self) -> bool:
        return self.training.overwrite

    @overwrite.setter
    def overwrite(self, value: bool) -> None:
        self.training.overwrite = value

    @property
    def max_length(self) -> int:
        return self.training.max_length

    @max_length.setter
    def max_length(self, value: int) -> None:
        self.training.max_length = value

    @property
    def num_epochs(self) -> int | None:
        return self.training.num_epochs if self.training.num_epochs is not None else self.training.epochs

    @num_epochs.setter
    def num_epochs(self, value: int | None) -> None:
        self.training.num_epochs = value
        if value is not None:
            self.training.epochs = value

    @property
    def freeze_backbone(self) -> bool:
        return self.training.freeze_backbone

    @freeze_backbone.setter
    def freeze_backbone(self, value: bool) -> None:
        self.training.freeze_backbone = value

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "TaxonomyConfig":
        """Load and validate taxonomy configuration from a YAML file."""
        path = Path(yaml_path)
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file not found at: {path.resolve()}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"YAML configuration at {path} must define a mapping/dictionary, got {type(data)}")

        return cls.model_validate(data)

