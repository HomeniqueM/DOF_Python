from enum import Enum 

class TrainingType(Enum):
    Binary = "Binária"
    KL = "Cinco classes KL"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))

    @classmethod
    def enum_to_int(cls,item):
        return list(map(lambda c: c.value, cls)).index(item.value)