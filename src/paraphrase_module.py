import pytorch_lightning as pl
import pandas as pd
from dataset import ParaphraseCustomDataset
from transformers import T5TokenizerFast
from torch.utils.data import DataLoader

class ParaphraseModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tokenizer: T5TokenizerFast,
        batch_size: int = 8,
        input_text_max_len: int = 128,
        target_text_max_len: int = 128,   
        
    ):
        super().__init__()
    
        self.train_df = train_df
        self.test_df = test_df
        
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.input_text_max_len = input_text_max_len
        self.target_text_max_len = target_text_max_len
    def setup(self, stage: str) -> None:
        self.train_dataset = ParaphraseCustomDataset(
            self.train_df,
        )
        
        self.test_dataset = ParaphraseCustomDataset(
            self.train_df,
        )
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=12
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=12
        )
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=12
        )
    