from dataclasses import dataclass
from model.model_type import ModelType


@dataclass(frozen=True)
class ModelInfo:
    type: ModelType
    equalize_images: bool
    horizontal_mirror: bool


def parse_model_info(text) -> ModelInfo:
    model_data = text.split("_")
    model_name = ['SVM', 'XGBoost', 'GoogleNet']
    model_bool = ['0', '1']
    
    # Valida se o nome do arquivo de cache repeita as regras 
    if model_data[1] not in  model_name and model_data[2][1] not in model_bool and model_data[3][1] not in model_bool:
        return

    return ModelInfo(ModelType(model_data[1]),
                     bool(int(model_data[2][1])),
                     bool(int(model_data[3][1])))
