from flax.training import train_state


class TrainState(train_state.TrainState):
    """A data-class storing model's parameters, optimizer and others
    """
    batch_stats: dict