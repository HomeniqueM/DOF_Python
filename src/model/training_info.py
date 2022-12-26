from dataclasses import dataclass
from model.model_info import ModelInfo
from model.training_type import TrainingType

@dataclass(frozen=True)
class TrainingInfo:
    model: ModelInfo
    training_type: TrainingType
    training_images_path: str