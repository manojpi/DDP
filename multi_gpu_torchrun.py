import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datautils import MyTrainDataset

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os


def ddp_setup():

    init_process_group(backend="nccl")

class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_data: DataLoader,
            optimizer: torch.optim.Optimizer,
            save_every: int,
            snapshot_path: str) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path

        if os.path.exists(self.snapshot_path):
            print("Snapshot of the model exists, so loading the snapshot")
            self._load_snapshot(self.snapshot_path)
        self.model = DDP(self.model, device_ids=[self.gpu_id])
    
    def _load_snapshot(self, snapshot_path):

        snapshot = torch.load(snapshot_path)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming model trainig from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()
    
    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0]) # TODO: Might be able to simplify this
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            self._run_batch(source, targets)

    def _save_checkpoint(self, epoch):
        snapshot = {}
        snapshot["MODEL_STATE"] = self.model.module.state_dict()
        snapshot["EPOCHS_RUN"] = epoch
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} |  Training checkpoint saved at {self.snapshot_path}")       

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

def initialize_train_objs():
    train_data = MyTrainDataset(2048)
    model = torch.nn.Linear(20, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    return train_data, model, optimizer

def initialize_dataloader(dataset: Dataset, batch_size: int):
    return DataLoader(
        dataset,
        batch_size,
        pin_memory=True, # pin_memory = page-locked memory, results in faster data copy between CPU -> GPU
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )

def main(save_every: int, total_epochs: int, snapshot_path: str = "snapshot.pt"):
    ddp_setup()
    dataset, model, optimizer = initialize_train_objs()
    train_data = initialize_dataloader(dataset, batch_size = 32)
    trainer = Trainer(model, train_data, optimizer, save_every, snapshot_path)
    trainer.train(total_epochs)
    destroy_process_group()

if __name__ == "__main__":
    import sys

    total_epochs = int(sys.argv[1])
    save_every = int(sys.argv[2])

    main(save_every, total_epochs)