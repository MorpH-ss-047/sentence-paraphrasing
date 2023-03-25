import config
import pandas as pd


class ParaphraseCustomDataset:
    def __init__(
        self,
        data: pd.DataFrame,
    ):
        self.data = data
        self.tokenizer = config.TOKENIZER
        self.input_max_len = config.MAX_LEN
        self.target_max_len = config.MAX_LEN

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index:int]

        input_text: str = data_row["input_text"]
        input_text = " ".join(input_text.split())

        # tokenizing the input text
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.input_max_len,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensor="pt",
        )

        # tokenizing the target text
        target_text: str = data_row["target_text"]
        target_text = " ".join(target_text.split())

        target_encoding = self.tokenizer(
            target_text,
            max_length=self.target_max_len,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensor="pt",
        )

        labels = target_encoding["input_ids"]
        labels[labels == 0] = -100

        return dict(
            input_text=input_text,
            target_text=data_row["target_text"],
            input_ids=input_encoding["input_ids"].flatten(),
            labels=labels.flatten(),
            decoder_attention_mask=target_encoding["attention_mask"].flatten(),
        )
