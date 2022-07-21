from __future__ import annotations

import os
import random
from datetime import datetime

import pytorch_metric_learning.utils.accuracy_calculator as AC
import pytorch_metric_learning.utils.logging_presets as LP
import torch
from pytorch_metric_learning import losses
from pytorch_metric_learning import miners
from pytorch_metric_learning import testers
from pytorch_metric_learning import trainers


def set_seed(seed: int = 42) -> None:
    """
    Set seed for all underlying (pseudo) random number sources.

    :param seed: seed to be used
    :return: None
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TrainTask:
    def __init__(
        self,
        train_dataset,
        model,
        batch_size=96,
        csv_folder="logs",
        num_workers=None,
        patience=3,
        k=None,
    ) -> None:
        self.train_dataset = train_dataset
        self.model = model

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.timestamp = self._get_timestamp()
        if num_workers is None:
            num_workers = os.cpu_count()
            num_workers = 1 if num_workers is None else num_workers
            num_workers = 8 if num_workers > 8 else num_workers
        # Set the loss function
        loss = losses.SupConLoss()

        # Set the mining function
        miner = miners.MultiSimilarityMiner()

        # Set other training parameters
        trunk = torch.nn.DataParallel(self.model).to(device)
        trunk_optimizer = torch.optim.AdamW(
            trunk.parameters(),
            lr=1e-4,
            weight_decay=0.01,
        )

        models = {"trunk": trunk}
        optimizers = {"trunk_optimizer": trunk_optimizer}
        loss_funcs = {"metric_loss": loss}
        trainer_cls = trainers.MetricLossOnly
        mining_funcs = {"tuple_miner": miner}
        accuracy_calculator = AC.AccuracyCalculator(k=k)

        self.log_folder = f"{csv_folder}/{self.timestamp}"
        record_keeper, _, _ = LP.get_record_keeper(self.log_folder)

        dataset_dict = {"val": train_dataset}
        model_folder = f"saved_models/{self.timestamp}"

        self.hooks = LP.HookContainer(
            record_keeper,
            record_group_name_prefix="metric_learn",
            primary_metric="mean_average_precision_at_r",
            validation_split_name="val",
            save_models=True,
        )

        # Create the tester
        self.tester = testers.GlobalEmbeddingSpaceTester(
            end_of_testing_hook=self.hooks.end_of_testing_hook,
            dataloader_num_workers=num_workers,
            accuracy_calculator=accuracy_calculator,
        )

        end_of_epoch_hook = self.hooks.end_of_epoch_hook(
            self.tester,
            dataset_dict,
            model_folder,
            test_interval=5,
            patience=patience,
        )

        self.trainer = trainer_cls(
            models,
            optimizers,
            batch_size,
            loss_funcs,
            mining_funcs,
            train_dataset,
            # sampler=sampler,
            dataloader_num_workers=num_workers,
            end_of_iteration_hook=self.hooks.end_of_iteration_hook,
            end_of_epoch_hook=end_of_epoch_hook,
        )

    def run(self, num_epochs=50):
        self.trainer.train(num_epochs=num_epochs)

    def acc_hist(self, skip=0):
        acc_hist = self.hooks.get_accuracy_history(
            self.tester,
            "val",
            return_all_metrics=True,
        )
        keys = acc_hist.keys()
        for idx, vs in enumerate(zip(*acc_hist.values())):
            if idx < skip:
                continue
            yield {k: v for k, v in zip(keys, vs)}

    def save_model(self, dst="model.pth"):
        torch.save(self.model.state_dict(), dst)
        return dst

    @staticmethod
    def _get_timestamp() -> int:
        return int(datetime.timestamp(datetime.now()) * 1000)
