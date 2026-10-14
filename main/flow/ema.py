"""Trainable-parameter EMA with FastVideo's per-outer-iteration update policy."""

from contextlib import contextmanager

import torch


class StudentEMA:
    def __init__(self, model, decay=.98):
        self.decay = float(decay)
        self.num_updates = 0
        self.shadow = self._snapshot(model)

    @staticmethod
    def _snapshot(model):
        return {name: parameter.detach().float().cpu().clone()
                for name, parameter in model.named_parameters() if parameter.requires_grad}

    @torch.no_grad()
    def update(self, model):
        # FastVideo reinitializes the shadow at the first training-step end.
        if self.num_updates == 0:
            self.shadow = self._snapshot(model)
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.shadow[name].mul_(self.decay).add_(parameter.detach().float().cpu(), alpha=1 - self.decay)
        self.num_updates += 1

    def state_dict(self):
        return dict(decay=self.decay, num_updates=self.num_updates,
                    shadow={name: value.clone() for name, value in self.shadow.items()})

    def load_state_dict(self, state):
        if state['decay'] != self.decay or state['shadow'].keys() != self.shadow.keys():
            raise ValueError('Student EMA configuration or parameter names differ from checkpoint')
        if any(state['shadow'][name].shape != value.shape for name, value in self.shadow.items()):
            raise ValueError('Student EMA parameter shapes differ from checkpoint')
        self.shadow = {name: value.detach().float().cpu().clone() for name, value in state['shadow'].items()}
        self.num_updates = int(state['num_updates'])

    @contextmanager
    def apply(self, model):
        saved = self._snapshot(model)
        try:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name in self.shadow:
                        parameter.copy_(self.shadow[name].to(parameter))
            yield model
        finally:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name in saved:
                        parameter.copy_(saved[name].to(parameter))
