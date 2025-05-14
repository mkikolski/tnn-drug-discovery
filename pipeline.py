from __future__ import annotations
from datetime import datetime

class Pipeline:
    def __init__(self, **initial_args):
        self.steps: list[dict] = []
        self.initial_args = initial_args

    def add_step(self, step: dict) -> Pipeline:
        self.steps.append(step)
        return self

    def run(self):
        next_args = self.initial_args
        for step in self.steps:
            print(f"[{datetime.now().isoformat()}] Running step: {step['name']}")
            next_args = step['function'](**next_args)
            print(f"[{datetime.now().isoformat()}] Step {step['name']} completed.")
