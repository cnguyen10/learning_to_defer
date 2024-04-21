import jax
from jax import numpy as jnp

import flax
from TrainState import TrainState
import orbax.checkpoint as ocp
import chex

import optax

from clu import metrics

import tensorflow_probability.substrates.jax as tfp

import mlx.data as dx

import mlflow

from pathlib import Path
import os
import json
import argparse
import logging
from tqdm import tqdm
import random

from PreActResNet import ResNet10
from ConvNet import ConvNet


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Parse input arguments')

    parser.add_argument('--experiment-name', type=str, default='Learning to defer')
    parser.add_argument('--run-description', type=str, default=None)

    parser.add_argument('--train-files', type=str, help='Delimited list of training files: "path/to/file1.json,path/to/file2.json"'
    )
    parser.add_argument('--train-groundtruth-file', type=str, help='Path to ground truth of training set')

    parser.add_argument('--test-files', type=str, help='Delimited list of testing files')
    parser.add_argument('--test-groundtruth-file', type=str, help='Path to ground truth of training set')

    parser.add_argument('--num-classes', type=int, help='Number of classes')
    parser.add_argument('--num-experts', type=int, default=2, help='Number of experts')

    parser.add_argument('--num-clusters', type=int, default=2, help='Number of expert clusters')

    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=100, help='The total number of epochs to run')

    parser.add_argument('--resume', dest='resume', action='store_true')
    parser.set_defaults(resume=False)

    parser.add_argument('--run-id', type=str, default=None, help='Run ID in MLFlow')

    parser.add_argument('--jax-platform', type=str, default='cpu', help='cpu, cuda or tpu')
    parser.add_argument('--device-id', type=int, help='Which GPU is used')
    parser.add_argument('--mem-frac', type=float, default=0.9, help='Percentage of GPU memory allocated for Jax')

    parser.add_argument('--prefetch-size', type=int, default=8)
    parser.add_argument('--num-threads', type=int, default=2)

    parser.add_argument('--tqdm', dest='tqdm', action='store_true')
    parser.add_argument('--no-tqdm', dest='tqdm', action='store_false')
    parser.set_defaults(tqdm=True)

    parser.add_argument('--train', dest='train_flag', action='store_true')
    parser.add_argument('--test', dest='train_flag', action='store_false')
    parser.set_defaults(train_flag=True)

    parser.add_argument('--tracking-uri', type=str, default='http://127.0.0.1:8080', help='MLFlow server')
    parser.add_argument('--artifact-path', type=str, help='Path to save model')

    args = parser.parse_args()

    # parse the list of files
    args.train_files = [i for i in args.train_files.split(',')]
    args.test_files = [i for i in args.test_files.split(',')]

    return args


def config_jax(mem_frac: float, platform: str) -> None:
    """configure the JAX environment variables

    Args:
        mem_frac: the percentage of GPU allocation for JAX
        platform: cuda or cpu
        device_id:
    """
    jax.config.update('jax_platforms', platform)

    # set jax memory allocation
    assert mem_frac < 1. and mem_frac > 0.
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(mem_frac)

    return None


def initialise_model(
    model_class: flax.linen.Module,
    num_classes: chex.Numeric,
    sample: chex.Array,
    num_training_samples: int,
    key: chex.PRNGKey
) -> tuple[TrainState, TrainState]:
    """initialise the parameters and optimiser of a model

    Args:
        sample: a sample from the dataset

    Returns:
        state:
    """
    model = model_class(num_classes=num_classes)
    logging.info(msg='The class of model is: {:s}'.format(model.__class__.__name__))

    lr_schedule_fn = optax.cosine_decay_schedule(
        init_value=args.lr,
        decay_steps=(args.num_epochs + 1) * (num_training_samples // args.batch_size + 1)
    )

    # pass dummy data to initialise model's parameters
    params = model.init(rngs=key, x=sample, train=False)

    # add L2 regularisation(aka weight decay)
    weight_decay = optax.masked(
        inner=optax.add_decayed_weights(
            weight_decay=0.0005,
            mask=None
        ),
        mask=lambda p: jax.tree_util.tree_map(lambda x: x.ndim != 1, p)
    )

    # define an optimizer
    tx = optax.chain(
        weight_decay,
        optax.sgd(learning_rate=lr_schedule_fn, momentum=0.9)
    )

    state = TrainState.create(
        apply_fn=model.apply,
        params=params['params'],
        batch_stats=params['batch_stats'],
        tx=tx
    )

    return state


def get_dataset(dataset_file: Path | str) -> dx._c.Buffer:
    """load a dataset from a JSON file

    Args:
        dataset_file: path to the JSON file

    Returns:
        dset: the dataset of interest
    """
    # load information from the JSON file
    with open(file=dataset_file, mode='r') as f:
        # load a list of dictionaries
        json_data = json.load(fp=f)

    data_dicts = [
        dict(
            file=data_dict['file'].encode('ascii'),
            label=data_dict['label']
        ) for data_dict in json_data
    ]

    # load image dataset without batching nor shuffling
    dset = (
        dx.buffer_from_vector(data=data_dicts)
        .load_image(key='file', output_key='image')
    )

    return dset


def get_dataset_multiple_labels(dataset_files: list[str | Path]) -> dx._c.Buffer:
    """load a dataset where labels are stored in many files
    NOTE: all files are assumed coming from the same dataset. Also, the orders of
    samples in those files are the same.

    Args:
        dataset_files:

    Returns:
        dset: each sample is a dictionary with the following keys:
            - file: path to file (in uint8)
            - label: 1-d array of integers
            - image:
    """
    assert len(dataset_files) > 1

    # initialise a list to store all the json data from all the provided datasets
    multiple_sets = []

    # load json data
    for dataset_file in dataset_files:
        with open(file=dataset_file, mode='r') as f:
            # load a list of dictionaries
            json_data = json.load(fp=f)

        # append each dataset
        multiple_sets.append(json_data)

    # list of dictionaries, each dictionary is a sample
    data_dicts = [
        dict(file=data_dict['file'].encode('ascii'), label=[]) \
            for data_dict in multiple_sets[0]
    ]

    # add additional labels
    for subset in multiple_sets:
        for data_dict, sample in zip(data_dicts, subset):
            # # verify if each is poiting to the same sample
            # assert data_dict['file'] == sample['file']

            # add the additional label into the dictionary of that sample
            data_dict['label'].append(sample['label'])

    # load image dataset without batching nor shuffling
    dset = (
        dx.buffer_from_vector(data=data_dicts)
        .load_image(key='file', output_key='image')
    )

    return dset


def prepare_dataset(dataset: dx._c.Buffer, shuffle: bool) -> dx._c.Buffer:
    """batch, shuffle and convert from uint8 to float32"""
    if shuffle:
        dset = dataset.shuffle()
    else:
        dset = dataset

    dset = (
        dset
        .to_stream()
        .batch(batch_size=args.batch_size)
        .key_transform(key='image', func=lambda x: x.astype('float32') / 255)
        .prefetch(prefetch_size=args.prefetch_size, num_threads=args.num_threads)
    )
    return dset


def get_coverage_selection(x: chex.Array, dfr_state: TrainState) -> chex.Array:
    """get multiple-hot vector representing the number of times the model is selected

    Args:
        x: a mini-batch of samples
        dfr_state:

    Returns:
        selections:
    """
    logits = prediction_step(x, dfr_state)
    selections = jnp.argmax(a=logits, axis=-1)
    selections = jax.nn.one_hot(x=selections, num_classes=len(args.train_files) + 1)
    
    return selections


@jax.jit
def E_step_single_sample(
    logits_p_z: chex.Array,
    logits_p_y__x_model: chex.Array,
    y_human: chex.Array,
    y: chex.Array
) -> chex.Scalar:
    # calculate prior of z
    log_p_z = jax.nn.log_softmax(x=logits_p_z, axis=-1)

    # region calculate p(y | x, z)
    # define distribution p(y | x, z = clf)
    p_y__x_model = tfp.distributions.Categorical(
        logits=jax.nn.log_softmax(x=logits_p_y__x_model, axis=-1)
    )
    log_prob_y__x_model = p_y__x_model.log_prob(value=y)

    # define distribution p(y | x, z = human)
    p_y__x_human = tfp.distributions.Categorical(
        probs=jax.nn.one_hot(x=y_human, num_classes=args.num_classes)
    )
    log_prob_y__x_human = p_y__x_human.log_prob(value=y)

    # log_prob_y__x_z = jnp.concatenate(
    #     arrays=(log_prob_y__x_human, log_prob_y__x_model),
    #     axis=0
    # )
    log_prob_y__x_z = jnp.append(arr=log_prob_y__x_human, values=log_prob_y__x_model)
    # endregion

    # calculate the posterior of z
    log_p_z__xy = log_prob_y__x_z + log_p_z  # up to the normalisation const.
    log_p_z__xy = log_p_z__xy - jax.nn.logsumexp(a=log_p_z__xy, axis=-1)
    p_z__xy = jnp.exp(log_p_z__xy)

    return p_z__xy


def E_step(
    x: chex.Array,
    y_human: chex.Array,
    y: chex.Array,
    clf_state: TrainState,
    dfr_state: TrainState
) -> chex.Scalar:
    """
    """
    logits_p_z_batch = prediction_step(x, dfr_state)
    logits_p_y__x_model_batch = prediction_step(x, clf_state)

    E_step_batch = jax.vmap(fun=E_step_single_sample, in_axes=(0, 0, 0, 0), out_axes=0)

    return E_step_batch(logits_p_z_batch, logits_p_y__x_model_batch, y_human, y)


@jax.jit
def M_step_batch(
    x: chex.Array,
    y: chex.Array,
    p_z__xy: chex.Array,
    clf_state: TrainState,
    dfr_state: TrainState
) -> tuple[TrainState, chex.Array, TrainState, chex.Array]:
    """perform the M-step

    Args:
        x: a mini-batch of input samples
        y_human: a mini-batch of annotated (int) labels
        y: a mini-batch of ground truth (int) labels
        p_z__xy: the posterior p(z | x, y)
        clf_state: the parameter of the classifier
        dfr_state: the parameter of the deferral function

    Returns:
        clf_state:
        clf_loss:
        dfr_state:
        dfr_loss:
    """
    def loss_on_data(
        params: flax.core.frozen_dict.FrozenDict,
        batch_stats: flax.core.frozen_dict.FrozenDict
    ) -> tuple[chex.Array, flax.core.frozen_dict.FrozenDict]:
        """calculate the loss related to the classifier:
        E_{p(z | x, y, theta_old, pi_old)} [ln p(y | x, z, theta)]

        Args:
            params: parameters of the classifier
            batch_stats: batch statistics of the classifier

        Returns:
            loss:
            batch_stats_new:
        """
        logits, batch_stats_new = clf_state.apply_fn(
            variables={'params': params, 'batch_stats': batch_stats},
            x=x,
            train=True,
            mutable=['batch_stats']
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits,
            labels=y
        )
        loss = loss * p_z__xy[:, -1]
        loss = jnp.mean(a=loss, axis=-1)

        return loss, batch_stats_new

    def loss_on_p_z(
        params: flax.core.frozen_dict.FrozenDict,
        batch_stats: flax.core.frozen_dict.FrozenDict
    ) -> tuple[chex.Array, flax.core.frozen_dict.FrozenDict]:
        """calculate the loss related to the deferral function
        E_{p(z | x, y, theta_old, pi_old)} [ln p(z | pi)]
        """
        logits, batch_stats_new = dfr_state.apply_fn(
            variables={'params': params, 'batch_stats': batch_stats},
            x=x,
            train=True,
            mutable=['batch_stats']
        )

        loss = optax.softmax_cross_entropy(logits=logits, labels=p_z__xy)
        loss = jnp.mean(a=loss, axis=-1)

        return loss, batch_stats_new

    # classifier
    clf_grad_value_fn = jax.value_and_grad(fun=loss_on_data, argnums=0, has_aux=True)
    (clf_loss, clf_batch_stats), clf_grads = clf_grad_value_fn(clf_state.params, clf_state.batch_stats)

    clf_state = clf_state.apply_gradients(grads=clf_grads)
    clf_state = clf_state.replace(batch_stats=clf_batch_stats['batch_stats'])

    # deferral function
    dfr_grad_value_fn = jax.value_and_grad(fun=loss_on_p_z, argnums=0, has_aux=True)
    (dfr_loss, dfr_batch_stats), dfr_grads = dfr_grad_value_fn(dfr_state.params, dfr_state.batch_stats)

    dfr_state = dfr_state.apply_gradients(grads=dfr_grads)
    dfr_state = dfr_state.replace(batch_stats=dfr_batch_stats['batch_stats'])

    return clf_state, clf_loss, dfr_state, dfr_loss


def train(
    dataset: dx._c.Buffer,
    clf_state: TrainState,
    dfr_state: TrainState
) -> tuple[TrainState, chex.Scalar, TrainState, chex.Scalar, chex.Scalar]:
    """training procedure

    Args:
        dataset:
        clf_state:
        dfr_state:

    Returns:
        clf_state:
        clf_loss: for monitoring/tracking purposes
        dfr_state:
        dfr_loss:
        coverage:
    """
    # prepare dataset for training
    dset = prepare_dataset(dataset=dataset, shuffle=True)

    # define metrics for monitoring purposes
    clf_loss_accum = metrics.Average(total=jnp.array(0.), count=jnp.array(0))
    dfr_loss_accum = metrics.Average(total=jnp.array(0.), count=jnp.array(0))
    coverage = metrics.Average(total=jnp.array(0.), count=jnp.array(0))

    for samples in tqdm(
        iterable=dset,
        desc='per epoch',
        total=len(dataset)//args.batch_size + 1,
        leave=False,
        position=2,
        disable=not args.tqdm
    ):
        x = jax.device_put(x=samples['image'], device=JAX_DEVICE)
        y_human = jax.device_put(x=samples['label'][:, :-1], device=JAX_DEVICE)
        y = jax.device_put(x=samples['label'][:, -1], device=JAX_DEVICE)

        # to monitor coverage on the training set
        selections = get_coverage_selection(x=x, dfr_state=dfr_state)
        coverage = metrics.Average.merge(
            self=coverage,
            other=metrics.Average.from_model_output(values=selections[:, -1:])
        )

        # E-step
        # calculate the posterior: p(z | x, y, theta_old, pi_old)
        p_z__xy = E_step(x, y_human, y, clf_state, dfr_state)

        # M-step
        clf_state, clf_loss, dfr_state, dfr_loss = M_step_batch(
            x,
            y,
            p_z__xy,
            clf_state,
            dfr_state
        )

        clf_loss_accum = metrics.Average.merge(
            self=clf_loss_accum,
            other=metrics.Average.from_model_output(values=clf_loss)
        )
        dfr_loss_accum = metrics.Average.merge(
            self=dfr_loss_accum,
            other=metrics.Average.from_model_output(values=dfr_loss)
        )


    return clf_state, clf_loss_accum.compute(), dfr_state, dfr_loss_accum.compute(), coverage.compute()


@jax.jit
def prediction_step(x: chex.Array, state: TrainState) -> chex.Array:
    """forward function without returning batch statistics

    Args:
        x: a batch of input samples
        state:

    Returns:
        logits:
    """
    logits, _ = state.apply_fn(
        variables={'params': state.params, 'batch_stats': state.batch_stats},
        x=x,
        train=False,
        mutable=['batch_stats']
    )
    return logits


def evaluate(
    dataset: dx._c.Buffer,
    clf_state: TrainState,
    dfr_state: TrainState
) -> tuple[chex.Scalar, chex.Scalar]:
    """Evaluate the model on the given dataset

    Args:
        dataset:
        clf_state:
        dfr_state:

    Returns:
        accuracy:
        coverage:
    """
    # batch the dataset
    dset = prepare_dataset(dataset=dataset, shuffle=False)

    # initialise metrics to monitor
    accuracy = metrics.Accuracy(total=jnp.array(0.), count=jnp.array(0))
    coverage = metrics.Average(total=jnp.array(0.), count=jnp.array(0))

    for samples in tqdm(
        iterable=dset,
        desc=' validate',
        total=len(dataset)//args.batch_size + 1,
        position=2,
        leave=False,
        disable=not args.tqdm
    ):
        x = jax.device_put(x=samples['image'], device=JAX_DEVICE)
        y_human = jax.device_put(x=samples['label'][:, :-1], device=JAX_DEVICE)
        y = jax.device_put(x=samples['label'][:, -1], device=JAX_DEVICE)

        y_human = jax.nn.one_hot(x=y_human, num_classes=args.num_classes)

        logits_model = prediction_step(x, clf_state)

        selections = get_coverage_selection(x=x, dfr_state=dfr_state)

        logits = jnp.concatenate(
            arrays=(y_human, logits_model[:, None, :]),
            axis=-2
        )
        logits = logits * selections[:, :, None]
        logits = jnp.sum(a=logits, axis=-2)

        # log metrics
        accuracy = metrics.Accuracy.merge(
            self=accuracy,
            other=metrics.Accuracy.from_model_output(logits=logits, labels=y)
        )
        coverage = metrics.Average.merge(
            self=coverage,
            other=metrics.Average.from_model_output(values=selections[:, -1])
        )

    return accuracy.compute(), coverage.compute()


def main() -> None:
    # load training and testing datasets
    dset_train = get_dataset_multiple_labels(
        dataset_files=args.train_files + [args.train_groundtruth_file]
    )
    dset_test = get_dataset_multiple_labels(
        dataset_files=args.test_files + [args.test_groundtruth_file]
    )

    # initialise a model
    clf_state = initialise_model(
        model_class=ConvNet,
        num_classes=args.num_classes,
        sample=jnp.expand_dims(a=dset_train[0]['image'] / 255, axis=0),
        num_training_samples=len(dset_train),
        key=jax.random.key(seed=random.randint(a=0, b=1_000))
    )
    dfr_state = initialise_model(
        model_class=ResNet10,
        num_classes=len(args.train_files) + 1,
        sample=jnp.expand_dims(a=dset_train[0]['image'] / 255, axis=0),
        num_training_samples=len(dset_train),
        key=jax.random.key(seed=random.randint(a=0, b=1_000))
    )

    # create a signature for mlflow model logging
    clf_signature = mlflow.models.ModelSignature(
        inputs=mlflow.types.Schema(
            inputs=[mlflow.types.TensorSpec(
                type=jnp.dtype(jnp.float32),
                shape=(-1,) + dset_train[0]['image'].shape
            )]
        ),
        outputs=mlflow.types.Schema(
            inputs=[mlflow.types.TensorSpec(
                type=jnp.dtype(jnp.float32),
                shape=(-1, args.num_classes)
            )]
        )
    )

    # dfr_signature = mlflow.models.ModelSignature(
    #     inputs=clf_signature.inputs,
    #     outputs=mlflow.types.Schema(
    #         inputs=[mlflow.types.TensorSpec(
    #             type=jnp.dtype(jnp.float32),
    #             shape=(-1, len(args.train_files) + 1)
    #         )]
    #     )
    # )

    # enable mlflow tracking
    with mlflow.start_run(
        log_system_metrics=True,
        description=args.run_description
    ) as mlflow_run:
        # append run id into the artifact path
        ckpt_dir = os.path.join(args.artifact_path, mlflow_run.info.run_id)

        # enable an orbax checkpoint manager to save model's parameters
        with ocp.CheckpointManager(
            directory=ocp.test_utils.erase_and_create_empty(directory=ckpt_dir),
            item_names=('clf_state', 'dfr_state'),
            options=ocp.CheckpointManagerOptions(
                save_interval_steps=10,
                max_to_keep=None,
                step_format_fixed_length=2
            )
        ) as mngr:
            # log hyper-parameters
            mlflow.log_params(
                params={key: args.__dict__[key] \
                    for key in args.__dict__ \
                        if isinstance(args.__dict__[key], (int, bool, str, float))}
            )

            for epoch_id in tqdm(
                iterable=range(args.num_epochs),
                desc='total',
                leave=True,
                position=1,
                disable=not args.tqdm
            ):
                # training
                clf_state, clf_loss, dfr_state, dfr_loss, train_coverage = train(
                    dataset=dset_train,
                    clf_state=clf_state,
                    dfr_state=dfr_state
                )
                mlflow.log_metrics(
                    metrics={
                        'loss/clf': clf_loss,
                        'loss/dfr': dfr_loss,
                        'coverage/train': train_coverage
                    },
                    step=epoch_id + 1
                )

                # evaluation
                pred_acc, test_coverage = evaluate(
                    dataset=dset_test,
                    clf_state=clf_state,
                    dfr_state=dfr_state
                )
                mlflow.log_metrics(
                    metrics={
                        'accuracy': pred_acc,
                        'coverage/test': test_coverage
                    },
                    step=epoch_id + 1
                )

                # save model's parameter
                save_or_skip = mngr.save(
                    step=epoch_id + 1,
                    args=ocp.args.Composite(
                        clf_state=ocp.args.StandardSave(clf_state),
                        dfr_state=ocp.args.StandardSave(dfr_state)
                    )
                )

                # logging model into mlflow
                if save_or_skip:
                    mngr.wait_until_finished()

                    # only log the classifier
                    mlflow.pyfunc.log_model(
                        artifact_path=os.path.join('mlruns', args.experiment_name),
                        python_model=lambda x: prediction_step(x, clf_state),
                        pip_requirements=['jax'],
                        artifacts={'clf_state': os.path.join(
                            ckpt_dir,
                            str(epoch_id + 1).zfill(mngr._options.step_format_fixed_length)
                        )},
                        signature=clf_signature
                    )

    return None


if __name__ == '__main__':
    # parse input arguments
    args = parse_arguments()

    # configure JAX environment variables
    config_jax(mem_frac=args.mem_frac, platform=args.jax_platform)

    JAX_DEVICE = jax.devices()[0]#[args.device_id]

    # enable MLFlow tracking
    os.environ['MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR'] = 'false'  # disable mlflow's tqdm
    mlflow.set_tracking_uri(uri=args.tracking_uri)
    mlflow.set_experiment(experiment_name=args.experiment_name)
    mlflow.set_system_metrics_sampling_interval(interval=60)
    mlflow.set_system_metrics_samples_before_logging(samples=1)

    # create a directory for storage (if not existed)
    if not os.path.exists(path=args.artifact_path):
        Path(args.artifact_path).mkdir(parents=True, exist_ok=True)

    main()
