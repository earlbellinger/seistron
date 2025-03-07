from nn_tools import TransformerBlock

class SurrogateModel():

    def sample(self):
        raise NotImplementedError

    def __call__(self):
        raise NotImplementedError


