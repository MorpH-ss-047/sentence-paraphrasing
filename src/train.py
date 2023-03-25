import config
from preprocessing import prepare_data
from paraphrase_module import ParaphraseModule

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import pandas as pd
from model import T5FineTuning
from sklearn.model_selection import train_test_split


def run():
    # loading the dataset
    df: pd.DataFrame = prepare_data.get_combined_dataset(
        google_paws=True, msrp=True, quora=True
    )

    # Splitting data into training and validation
    train_df, test_df = train_test_split(
        df,
        test_size=0.1,
        random_state=42,
    )

    # resetting index because DataLoader will use the indexes to pass
    # batches of data to model
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    data_module = ParaphraseModule(
        train_df=train_df,
        test_df=test_df,
        tokenizer=config.TOKENIZER,
        batch_size=config.TRAIN_BATCH_SIZE,
    )

    # Hyper parameters
    model_args = dict(
        model_name_or_path="t5-base",
        tokenizer_name_or_path="t5-base",
        max_seq_length=config.MAX_LEN,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        adam_epsilon=config.ADAM_EPSILON,
        warmup_steps=config.WARMUP_STEPS,
        train_batch_size=config.TRAIN_BATCH_SIZE,
        eval_batch_size=config.VALID_BATCH_SIZE,
        num_train_epochs=config.EPOCHS,
        gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
        n_gpu=1,
        early_stop_callback=config.EARLY_STOPPING,
        fp_16=False,
        opt_level="O1",
        max_grad_norm=1.0,
        seed=42,
    )

    # model
    model = T5FineTuning(model_args)

    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=config.DIRPATH,
    #     filename=config.FILENAME,
    #     save_top_k=1,
    #     verbose=True,
    #     monitor="val_loss",
    #     mode="min",
    # )

    logger = TensorBoardLogger("tb_logs", name="paraphrase")

    #  training starts here
    trainer = pl.Trainer(
        accelerator="gpu",
        logger=logger,
        max_epochs=config.EPOCHS,
        gpus=1,
        enable_progress_bar=True,
    )
    trainer.fit(model, data_module)
    


if __name__ == "__main__":
    run()
