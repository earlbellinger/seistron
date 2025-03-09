import orbax.checkpoint as ocp
import os


# TODO: Figure out type hint for state
def save_model(name, state, path=None):
    checkpoint = {
        'state': state,
    }
    if path is not None:
        dir_path = os.path.abspath(f'{path}/{name}')
    else:
        dir_path = os.path.abspath(f'./checkpoints/{name}')
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(dir_path, checkpoint)


def load_model(name, path=None):
    if path is not None:
        dir_path = os.path.abspath(f'{path}/{name}')
    else:
        dir_path = os.path.abspath(f'./checkpoints/{name}')
    checkpointer = ocp.PyTreeCheckpointer()
    return checkpointer.restore(dir_path)
