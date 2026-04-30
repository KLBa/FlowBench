
from ptlflow.models.raft import RAFT
from ptlflow.utils.registry import register_model

@register_model
class my_very_own_raft(RAFT):
    pass
