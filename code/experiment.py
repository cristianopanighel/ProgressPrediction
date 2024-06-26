import os
import statistics
import torch
import wandb

from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict

def get_device() -> torch.device:
    # if torch.backends.mps.is_available():
    #    device = "mps"
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return torch.device(device)

class Experiment:
    def __init__(
        self,
        network: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler,
        trainloader: DataLoader,
        testloader: DataLoader,
        train_fn,
        max_length: int,
        experiment_path: str,
        result: Dict,
    ) -> None:
        self.device = get_device()
        self.network = network.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.trainloader = trainloader
        self.testloader = testloader
        self.train_fn = train_fn
        self.max_length = max_length
        self.experiment_path = experiment_path
        self.result = result

        if experiment_path:
            os.makedirs(experiment_path, exist_ok=True)

    def print(self) -> None:
        print("--- Network ---")
        print(self.network)
        print("--- Datasets ---")
        print(
            f"Train {self.trainloader.dataset.splitfile} - {len(self.trainloader.dataset)} ({len(self.trainloader)})")
        if len(self.trainloader.dataset.lengths) > 1:
            print(
                f'- {statistics.mean(self.trainloader.dataset.lengths)} / {statistics.stdev(self.trainloader.dataset.lengths)}')
            print(
                f'- {min(self.trainloader.dataset.lengths)} / {max(self.trainloader.dataset.lengths)}')
        print(
            f"Test {self.testloader.dataset.splitfile} - {len(self.testloader.dataset)} ({len(self.testloader)})")
        if len(self.testloader.dataset.lengths) > 1:
            print(
                f'- {statistics.mean(self.testloader.dataset.lengths)} / {statistics.stdev(self.testloader.dataset.lengths)}')
            print(
                f'- {min(self.testloader.dataset.lengths)} / {max(self.testloader.dataset.lengths)}')
        print("--- Optimizer & Scheduler ---")
        print(self.optimizer)
        print(self.scheduler)

    def run(self, iterations: int, log_every: int, test_every: int) -> None:
        iteration = 0
        done = False
        train_result, test_result = self.result.copy(), self.result.copy()
        while not done:
            for batch in self.trainloader:
                batch_result = self.train_fn(
                    self.network,
                    self.criterion,
                    batch,
                    self.max_length,
                    self.device,
                    optimizer=self.optimizer,
                )
                self._add_result(train_result, batch_result)

                if iteration % log_every == 0 and iteration > 0:
                    train_result = self._log(train_result, iteration, "train")
                if iteration % test_every == 0 and iteration > 0:
                    self.network.eval()
                    with torch.no_grad():
                        for batch in self.testloader:
                            batch_result = self.train_fn(
                                self.network, self.criterion, batch, self.max_length, self.device
                            )
                            self._add_result(test_result, batch_result)
                    test_result = self._log(test_result, iteration, "test")
                    if self.experiment_path:
                        model_path = os.path.join(
                            self.experiment_path, f"model_{iteration}.pth"
                        )
                        torch.save(self.network.state_dict(), model_path)
                    self.network.train()

                iteration += 1
                if iteration > iterations:
                    done = True
                    break
                self.scheduler.step()

    def eval(self) -> None:
        self.network.eval()
        test_result = self.result.copy()
        with torch.no_grad():
            for batch in tqdm(self.testloader):
                batch_result = self.train_fn(
                    self.network, self.criterion, batch, self.max_length, self.device
                )
                self._add_result(test_result, batch_result)
        for key in test_result:
            if key == "count":
                continue
            print(f'{key}: {test_result[key] / test_result["count"]}')
        self.network.train()

    def save(self, save_dir: str) -> None:
        self.network.eval()
        criterion = nn.L1Loss(reduction='sum')
        l2_criterion = nn.MSELoss(reduction = 'sum')
        total_loss, l2_loss, count = 0, 0, 0
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.testloader)):
                progress = self.train_fn(
                    self.network,
                    self.criterion,
                    batch,
                    self.max_length,
                    self.device,
                    return_results=True,
                )
                total_loss += criterion(progress * 100, batch[-1] * 100).item()
                l2_loss += l2_criterion(progress, batch[-1]).item()
                progress = torch.flatten(progress).tolist()
                count += len(progress)
                txt = '\n'.join(map(str, progress))
                with open(f'./data/{save_dir}/{batch[0][0].replace("/", "_")}.txt', 'w+') as f:
                    # is i the index of the image, of the folder or a simple counter that start from 0?
                    f.write(txt)
        print("L1 loss")
        print(total_loss / count)
        print("L2 loss")
        print(l2_loss / count)
        self.network.train()

    @staticmethod
    def _add_result(result: Dict, batch_result: Dict) -> None:
        for key in result:
            if key in batch_result:
                result[key] += batch_result[key]

    def _log(self, result: Dict, iteration: int, prefix: str) -> Dict:
        log = {
            f"{prefix}_{key}": result[key] / result["count"]
            for key in result
            if key != "count"
        }
        log[f"{prefix}_count"] = result["count"]
        log["iteration"] = iteration
        wandb.log(log)
        return self.result.copy()
