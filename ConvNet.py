from flax import linen as nn

from typing import Any, Optional
import chex


class ConvNet(nn.Module):
    num_classes: int
    train: bool | None = None

    @nn.compact
    def __call__(self, x: chex.Array, train: Optional[bool] = None) -> Any:
        train = nn.merge_param(name='train', a=self.train, b=train)

        out = nn.Conv(features=32, kernel_size=(3, 3), strides=1, padding=1, use_bias=False)(inputs=x)
        out = nn.BatchNorm(use_running_average=not train)(x=out)
        out = nn.relu(x=out)

        out = nn.Conv(features=32, kernel_size=(3, 3), strides=1, padding=1, use_bias=False)(inputs=out)
        out = nn.BatchNorm(use_running_average=not train)(x=out)
        out = nn.relu(x=out)

        out = nn.Conv(features=32, kernel_size=(3, 3), strides=1, padding=1, use_bias=False)(inputs=out)
        out = nn.BatchNorm(use_running_average=not train)(x=out)
        out = nn.relu(x=out)

        # out = nn.Conv(features=32, kernel_size=(3, 3), strides=1, padding=1, use_bias=False)(inputs=out)
        # out = nn.BatchNorm(use_running_average=not train)(x=out)
        # out = nn.relu(x=out)

        out = out.mean(axis=(1, 2))

        out = nn.Dense(features=self.num_classes)(inputs=out)

        return out