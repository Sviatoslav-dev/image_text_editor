from enum import Enum, auto


class ImageAction(Enum):
    ReplaceText = auto()
    RemoveText = auto()
    CopyText = auto()
