"""Helper functions for running experiments and logging to Weights and Biases)."""

import inspect
import logging
import math
import pprint
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from source.library.architectures import FullyConnectedNN, ConvNet2L
from source.library.pytorch_helpers import EarlyStopping, calculate_average_loss


def get_available_device() -> str:
    """Returns the device based on the system configuration."""
    if torch.cuda.is_available():
        return torch.device('cuda').type
    if torch.backends.mps.is_available():
        return torch.device('mps').type  # https://pytorch.org/docs/stable/notes/mps.html
    return torch.device('cpu').type


def get_data(architecture: str):  # noqa
    """Function is required by and called from `model_pipeline()`."""
    assert architecture in ['FC', 'CNN'], f"Architecture {architecture} not supported."
    x, y = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
    x = torch.tensor(x.values, dtype=torch.float32)
    x = transform_data(x)
    y = torch.tensor(y.astype(int).values, dtype=torch.long)

    if architecture == 'CNN':
        # Reshape data to have channel dimension
        # MNIST images are 28x28, so we reshape them to [batch_size, 1, 28, 28]
        x = x.reshape(-1, 1, 28, 28)

    # 80% train; 10% validation; 10% test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=42)
    logging.info(f"Training set  : X-{x_train.shape}, y-{y_train.shape}")
    logging.info(f"Validation set: X-{x_val.shape}, y-{y_val.shape}")
    logging.info(f"Test set      : X-{x_test.shape}, y-{y_test.shape}")
    return x_train, x_val, x_test, y_train, y_val, y_test


def transform_data(x: torch.tensor) -> torch.tensor:
    """Transforms the data (e.g. normalizes generic x data from 0 to 1)."""
    # Normalize the tensor
    x_min = x.min()
    x_max = x.max()
    normalized_x = (x - x_min) / (x_max - x_min)
    assert normalized_x.min() == 0
    assert normalized_x.max() == 1
    return normalized_x


def model_pipeline(config: dict | None = None) -> nn.Module:
    """
    Builds the pipeline, data loaders, and model, and trains/tests. Results are logged to Weights
    and Biases.

    NOTE: This function requires `get_data()` function to be dfined, which should return the
    training, validation, and test data in the form of (x_train, x_val, x_test, y_train, y_val,
    y_test).

    Args:
        config: A dictionary of configuration parameters. If None, the configuration will be
            loaded from Weights and Biases (which implies a 'sweep' is running).
    """
    # if no config is provided, a sweep is running, and we will get the config from wandb.config
    project = config.pop('project') if config else None
    tags = config.pop('tags', None) if config else None
    notes = config.pop('notes', None) if config else None
    with wandb.init(project=project, config=config, tags=tags, notes=notes):
        config = wandb.config
        pprint.pprint(config)
        x_train, x_val, x_test, y_train, y_val, y_test = get_data(architecture=config.architecture)
        train_loader = make_loader(x_train, y_train, batch_size=config.batch_size)
        validation_loader = make_loader(x_val, y_val, batch_size=config.batch_size)
        model = make_model(input_size=28*28, output_size=10, config=config)
        criterion = nn.CrossEntropyLoss()
        optimizer_creator = make_optimizer(optimizer=config.optimizer, model=model)
        train(
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            criterion=criterion,
            optimizer_creator=optimizer_creator,
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            device=config.device,
            early_stopping_patience=config.early_stopping_patience,
            early_stopping_delta=config.early_stopping_delta,
            early_stopping_delta_type=config.early_stopping_delta_type,
            num_reduce_learning_rate=config.num_reduce_learning_rate,
        )
        evaluate(
            model=model,
            train_loader=train_loader,
            validation_loader=validation_loader,
            x_test=x_test,
            y_test=y_test,
            criterion=criterion,
            device=config.device,
        )
    return model


def make_loader(x: torch.tensor, y: torch.tensor, batch_size: int) -> DataLoader:
    """Make a DataLoader from a given dataset."""
    return DataLoader(
        dataset=TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
    )


def _get_parameters(func: callable) -> list:
    """Get the list of parameters of a function."""
    return [k for k in inspect.signature(func).parameters if k != 'self']


def make_model(input_size: int, output_size: int, config: dict) -> nn.Module:
    """Make a model based on the architecture."""
    architecture = config['architecture']
    device = config['device']
    assert architecture in ['FC', 'CNN'], f"Unknown model type: {architecture}"
    assert device
    if architecture == 'FC':
        model_parameters = _get_parameters(FullyConnectedNN.__init__)
        model = FullyConnectedNN(
            input_size=input_size,
            output_size=output_size,
            **{k: v for k, v in config.items() if k in model_parameters},
        )
    elif architecture == 'CNN':
        dimension = int(math.sqrt(input_size))
        model = ConvNet2L(
            dimensions=(dimension, dimension),
            output_size=output_size,
            **{k: v for k, v in config.items() if k in _get_parameters(ConvNet2L.__init__)},
        )
    else:
        raise ValueError(f"Unknown model type: {architecture}")
    model = model.to(device)
    print(model)
    return model


def make_optimizer(optimizer: str, model: nn.Module) -> callable:
    """
    Make an optimizer based on the optimizer name. A function is returned that creates the
    optimizer, so that we can easily change the learning rate during training.
    """
    assert optimizer in ['adam', 'sgd'], f"Unknown optimizer: {optimizer}"
    if optimizer == 'adam':
        return lambda lr: torch.optim.Adam(model.parameters(), lr=lr)
    if optimizer == 'sgd':
        return lambda lr: torch.optim.SGD(model.parameters(), lr=lr)
    raise ValueError(f"Unknown optimizer: {optimizer}")


def train(  # noqa: PLR0915
        model: nn.Module,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        criterion: callable,
        optimizer_creator: callable,
        epochs: int,
        learning_rate: float,
        device: str,
        early_stopping_patience: int = 3,
        early_stopping_delta: float = 0.05,
        early_stopping_delta_type: str = 'relative',
        num_reduce_learning_rate: int = 5,
        log_wandb: bool = True,
        ) -> None:
    """
    Trains the model for the number of epochs specified in the config. Uses early stopping to
    prevent overfitting. Takes multiple learning rates and if early stopping is triggered, the
    learning rate is reduced and training is continued until no learning rates remain.

    - num_reduce_learning_rate: The number of times to reduce the learning rate before stopping.
        The learning rate is reduced by a factor of 2 each time early stopping is triggered.
        The model is reloaded with the previous best state and training continues with the new
        learning rate. If the number of times the learning rate has been reduced equals
        num_reduce_learning_rate, training stops.
    """
    logging.info(f"Training on {device}; epochs: {epochs}; learning rate: {learning_rate}")
    model.train()
    if log_wandb:
        # Tell wandb to watch what the model gets up to: gradients, weights, and more!
        wandb.watch(model, criterion, log='all', log_freq=20)

    # Run training and track with wandb
    example_ct = 0  # number of examples seen
    log_interval = 30 # i.e. every 30 batches
    total_batches = len(train_loader)
    log_interval = max(1, math.floor(total_batches / log_interval))

    early_stopping = EarlyStopping(
        model=model,
        patience=early_stopping_patience,
        delta=early_stopping_delta,
        delta_type=early_stopping_delta_type,
        verbose=True,
    )
    stop_count = 0
    optimizer = optimizer_creator(lr=learning_rate)
    if log_wandb:
        wandb.log({'learning_rate': learning_rate})  # only log initial learning rate
    for epoch in tqdm(range(epochs)):
        logging.info(f"Epoch: {epoch} | Learning Rate: {learning_rate:.3f}")
        running_training_loss = 0
        total_train_samples = 0
        for batch_index, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)  # noqa: PLW2901
            # ➡ Forward pass
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            # ⬅ Backward pass & optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            example_ct += len(x_batch)
            # weighted average of the training loss
            running_training_loss += loss.item() * x_batch.shape[0]
            total_train_samples += x_batch.shape[0]
            # Report metrics every X batches
            if batch_index % log_interval == 0:
                avg_training_loss = running_training_loss / total_train_samples
                running_training_loss = 0
                total_train_samples = 0
                model.eval()
                average_validation_loss = calculate_average_loss(
                    data_loader=validation_loader, model=model, loss_func=criterion, device=device,
                )
                if log_wandb:
                    wandb.log(
                        {
                            'epoch': epoch,
                            'step_learning_rate': learning_rate,
                            'step_training_loss': avg_training_loss,
                            'step_validation_loss': average_validation_loss,
                        },
                        step=example_ct,
                    )
                logging.info(
                    f"Epoch: {epoch} | Learning Rate: {learning_rate:.6f}: "
                    f"Avg Training/Validation Loss after {example_ct:,} examples: "
                    f"{avg_training_loss:.3f} | {average_validation_loss:.3f}",
                )
                model.train()

        model.eval()
        average_validation_loss = calculate_average_loss(
            data_loader=validation_loader, model=model, loss_func=criterion, device=device,
        )
        model.train()
        if early_stopping(average_validation_loss):
            logging.info("Early stopping. Loading previous best state.")
            # we have stopped training (for this learning rate), load the previous best state
            model.load_state_dict(early_stopping.best_state)
            # if we have more learning rates, reset the optimizer and early stopping and
            # continue training
            if stop_count < num_reduce_learning_rate:
                logging.info(f"Reducing learning rate: {learning_rate} -> {learning_rate / 2}")
                learning_rate /= 2
                optimizer = optimizer_creator(lr=learning_rate)
                early_stopping.reset()
                stop_count += 1
            else:
                break

    if not early_stopping.is_stopped:
        # we have finished training, but early stopping was not triggered, so we need to load the
        # best state, which may or may not be the current state
        # e.g. we may have stopped training because we are at the limit of the number of epochs,
        # but either A) the validation loss is still decreasing, or B) the early stopping counter
        # has not yet reached the patience limit
        logging.info("Training completed without early stopping. Loading previous best state.")
        model.load_state_dict(early_stopping.best_state)

    if log_wandb:
        wandb.log({
            'best_validation_loss': early_stopping.lowest_loss,
            'best_epoch': early_stopping.best_index,
        })
        # Save the model in the exchangeable ONNX format
        torch.onnx.export(model, x_batch.to(device) , 'model.onnx')
        wandb.save('model.onnx')
    logging.info(f"Best validation loss: {early_stopping.lowest_loss:.3f}")
    logging.info(f"Best early stopping index/epoch: {early_stopping.best_index}")


def predict(model: nn.Module, x: torch.tensor, device: str, probs: bool = False) -> torch.tensor:
    """
    Predict the class for a given input. Returns a tensor of predictions. To convert to a numpy
    array, use `predict(model, x, device).cpu().numpy()`.
    """
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        outputs = model(x)
        if probs:
            return torch.softmax(outputs.data, dim=1)
        return torch.argmax(outputs.data, dim=1)


def evaluate(
        model: nn.Module,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        x_test: torch.tensor,
        y_test: torch.tensor,
        criterion: callable,
        device: str,
        log_wandb: bool = True) -> None:
    """Tests the model on the test set. Logs the accuracy to the console and to wandb."""
    model.eval()
    avg_training_loss = calculate_average_loss(
        data_loader=train_loader, model=model, loss_func=criterion, device=device,
    )
    logging.info(f"Final Average Loss on training set: {avg_training_loss:.3f}")

    avg_validation_loss = calculate_average_loss(
        data_loader=validation_loader, model=model, loss_func=criterion, device=device,
    )
    logging.info(f"Final Average Loss on validation set: {avg_validation_loss:.3f}")

    avg_test_loss = calculate_average_loss(
        data_loader=DataLoader(
            dataset=TensorDataset(x_test, y_test),
            batch_size=1000,
            shuffle=False,
        ),
        model=model,
        loss_func=criterion,
        device=device,
    )
    logging.info(f"Final Average Loss on test set: {avg_test_loss:.3f}")

    if log_wandb:
        wandb.log({
            'training_loss': avg_training_loss,
            'validation_loss': avg_validation_loss,
            'test_loss': avg_test_loss,
        })

    test_predictions = predict(model=model, x=x_test, device=device).cpu().numpy()
    test_labels = y_test.cpu().numpy()
    plot_misclassified_sample(num_images=30, images=x_test, predictions=test_predictions, labels=test_labels, log_wandb=log_wandb)  # noqa
    plot_heatmap(predictions=test_predictions, labels=test_labels, log_wandb=log_wandb)

    # for each class, calculate the accuracy metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_true=test_labels, y_pred=test_predictions)  # noqa
    plot_scores(precision, recall, f1, log_wandb=log_wandb)
    if log_wandb:
        score_table = wandb.Table(columns=["class", "precision", "recall", "f1"])
        for i in range(10):
            score_table.add_data(str(i), precision[i], recall[i], f1[i])
        wandb.log({"score_table": score_table})

    # compute scores for the entire dataset based on the test data
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true=test_labels,
        y_pred=test_predictions,
        average='weighted',
    )
    accuracy = np.mean(test_predictions == test_labels)
    logging.info(f"Weighted Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    logging.info(f"Accuracy: {accuracy:.3f}")
    if log_wandb:
        wandb.log({
            'weighted_precision': precision,
            'weighted_recall': recall,
            'weighted_f1': f1,
            'accuracy': accuracy,
        })


def plot_misclassified_sample(
        num_images: int,
        images: torch.tensor,
        predictions: np.array,
        labels: np.array,
        log_wandb: bool) -> None:
    """Plot a sample of the misclassified images."""
    fig, ax = plt.subplots(nrows=num_images // 5, ncols=5, sharex=True, sharey=True)
    ax = ax.flatten()
    mismatched_indexes = np.where(predictions != labels)[0]
    rows = np.random.choice(mismatched_indexes, size=num_images, replace=False)  # noqa: NPY002
    for i, row in enumerate(rows):
        # img = X_test[row].cpu().numpy().reshape(28, 28)
        img = images[row].cpu().numpy().reshape(28, 28)
        ax[i].imshow(img, cmap='Greys')
        title_color = 'red' if predictions[row] != labels[row] else 'black'
        ax[i].set_title(f'P:{predictions[row]} - A:{labels[row]}', color=title_color)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    if log_wandb:
        wandb.log({'sample-misclassified': wandb.Image(fig)})
        return None
    return fig


def plot_heatmap(predictions: np.array, labels: np.array, log_wandb: bool) -> None:
    """Plot a heatmap of the misclassified samples."""
    # create a heatmap of misclassified samples
    cm = confusion_matrix(labels, predictions)
    # remove the diagonal values (correct predictions) for better visualization
    np.fill_diagonal(cm, 0)
    fig = plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Count of Misclassified Samples by Class')
    if log_wandb:
        wandb.log({'count-misclassified': wandb.Image(fig)})
        return None
    return fig


def plot_scores(precision: list, recall: list, f1: list, log_wandb: bool) -> None:
    """Plot the precision, recall, and f1 scores for each class."""
    # create a bar plot
    x = range(10)
    width = 0.2
    fig, ax = plt.subplots()
    _ = ax.bar(x, precision, width, label='Precision')
    _= ax.bar([i + width for i in x], recall, width, label='Recall')
    _ = ax.bar([i + 2 * width for i in x], f1, width, label='F1')
    # add labels, title, and legend
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Accuracy Metrics by Class')
    ax.set_xticks([i + width for i in x])
    ax.set_xticklabels(range(10))
    ax.legend()
    # find the minimum and maximum score values (from precision, recall, and f1 lists) and set the
    # y limits slightly wider to make the plot easier to read
    ymin = min(*precision, *recall, *f1)
    ymax = max(*precision, *recall, *f1)
    ax.set_ylim([ymin - 0.03, min(ymax + 0.03, 1)])
    if log_wandb:
        wandb.log({'scores': wandb.Image(fig)})
        return None
    return fig
