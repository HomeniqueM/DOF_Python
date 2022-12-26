from enum import Enum 

class ModelType(Enum):
    XGBOOST = "XGBOOST"
    GOOGLENET = "GoogleNet"
    SVM = "SVM"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))

    @classmethod
    def enum_to_int(cls,item):
        return list(map(lambda c: c.value, cls)).index(item.value)