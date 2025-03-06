from abc import ABC, abstractmethod


class Emulator(ABC):

    @abstractmethod
    def sample(self):
        raise NotImplementedError

    @abstractmethod
    def __call__(self):
        raise NotImplementedError


class MLP(Emulator):

    def __init__(self):
        pass

    def sample(self):
        pass

    def __call__(self):
        pass


class Transformer(Emulator):

    def __init__(self):
        pass

    def sample(self):
        pass

    def __call__(self):
        pass

