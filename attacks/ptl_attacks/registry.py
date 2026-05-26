import sys

import lightning.pytorch as pl
from typing import Type

from .base_attack.base_attack import BaseAttack

_attacks_dict = {}

class RegisteredAttack(pl.LightningModule):
    pass

def register_attack(attack_class: Type[BaseAttack]) -> Type[BaseAttack]:
    # lookup containing module
    attack_dir = ".".join(attack_class.__module__.split(".")[:-1])
    mod = sys.modules[attack_dir]
    # add attack to __all__ in module
    attack_name = attack_class.__name__
    if hasattr(mod, "__all__"):
        mod.__all__.append(attack_name)
    else:
        mod.__all__ = [attack_name]  # type: ignore

    _attacks_dict[attack_class.__name__] = attack_class
    registered_class = type(attack_class.__name__, (attack_class, RegisteredAttack), {})
    registered_class.__module__ = attack_class.__module__
    return registered_class
