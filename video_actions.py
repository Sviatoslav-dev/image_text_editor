from enum import Enum, auto


class VideoAction(Enum):
    ReplaceText = auto()
    RemoveText = auto()
    CopyText = auto()
    FindText = auto()
