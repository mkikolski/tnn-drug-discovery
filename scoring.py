from __future__ import annotations
from typing import Callable
import os


class Scoring:
    def __init__(self, args: dict = None):
        self.components: list[tuple[Callable, float, tuple]] = []
        self.args = args if args else {}

    def set_args(self, args: dict):
        self.args = args

    def add_python_callable(self, func: Callable, weight: float = 1.0, args: tuple = ()) -> Scoring:
        self.components.append((func, weight, args))
        return self

    def add_external_callable(self, func: str, parse_result: Callable, weight: float = 1.0, args: tuple = ()) -> Scoring:
        def external_func(*args):
            instruction = func.format(args)
            result = os.popen(instruction).read()
            return parse_result(result)

        self.components.append((external_func, weight, args))
        return self

    def compute(self) -> float:
        total_score = 0.0
        total_weight = 0.0
        for func, weight, args in self.components:
            score = func(*(self.args[arg] for arg in args))
            total_score += score * weight
            total_weight += weight
        return total_score / total_weight if total_weight > 0 else 0.0
