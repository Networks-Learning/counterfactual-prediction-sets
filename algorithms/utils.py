from enum import Enum

class Status(Enum):
    SEARCH = 0
    DONE = 1

class Search(Enum):
    PIVOT = 0
    UPPER = 1
    LOWER = 2
    DONE = 3