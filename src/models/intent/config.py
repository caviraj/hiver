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
