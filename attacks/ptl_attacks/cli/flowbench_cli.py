
from typing import Optional, Type
from ptlflow.utils.lightning.ptlflow_cli import PTLFlowCLI

class FlowBenchCLI(PTLFlowCLI):
    
    def __init__(self, 
        attack_class: Optional[Type[object]] = None,
        **kwargs
    ):
        self._attack_class = attack_class
        super().__init__(**kwargs)

    def add_arguments_to_parser(self, parser):
        super().add_arguments_to_parser(parser)
        assert self._attack_class is not None
        print(self._attack_class)
        parser.add_argument(
            "--attack_obj",
            default="none",
            type=self._attack_class,
            help="Attack to be used.",
        )

    def instantiate_classes(self) -> None:
        super().instantiate_classes()
        
        self.attack = self._get(self.config_init, "attack_obj")
