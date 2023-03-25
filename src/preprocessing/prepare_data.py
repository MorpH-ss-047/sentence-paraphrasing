import pandas as pd


# DATASETS
GOOGLE_DATA = "../data/google/train.tsv"
MSRP_DATA = "../data/msrp/msr_paraphrase_train.txt"
PARABANK_DATA = "../data/parabank/parabank_5m.tsv"
QUORA_DATA = "../data/quora/quora_duplicate_questions.tsv"


def _load_data(path, input_column, target_column, label_column):
    df = pd.read_csv(path, sep="\t", on_bad_lines="skip").astype(str)
    df = df.loc[df[label_column] == "1"]
    df = df.rename(columns={input_column: "input_text", target_column: "target_text"})
    df = df[["input_text", "target_text"]]

    return df


def _clean_unnecessary_spaces(out_string):
    out_string = str(out_string)
    out_string = " ".join(out_string.split())
    out_string = (
        out_string.replace(" .", ".")
        .replace(" ?", "?")
        .replace(" !", "!")
        .replace(" ,", ",")
        .replace(" ' ", "'")
        .replace(" n't", "n't")
        .replace(" 'm", "'m")
        .replace(" 's", "'s")
        .replace(" 've", "'ve")
        .replace(" 're", "'re")
    )
    return out_string


## load datasets
def get_google_dataset() -> pd.DataFrame:
    google_df = _load_data(GOOGLE_DATA, "sentence1", "sentence2", "label")
    return google_df


def get_quora_dataset() -> pd.DataFrame:
    quora_df = _load_data(QUORA_DATA, "question1", "question2", "is_duplicate")
    return quora_df


def get_msrp_dataset() -> pd.DataFrame:
    msrp_df = _load_data(MSRP_DATA, "#1 String", "#2 String", "Quality")
    return msrp_df




def get_combined_dataset(
    google_paws: bool = False,
    msrp: bool = False,
    quora: bool = False,
    randomize: bool = True,
) -> pd.DataFrame:
    df = pd.DataFrame()
    if google_paws:
        google_df = get_google_dataset()
        df = pd.concat([df, google_df])
    if msrp:
        msrp_df = get_msrp_dataset()
        df = pd.concat([df, msrp_df])
    if quora:   
        quora_df = get_quora_dataset()
        df = pd.concat([df, quora_df])
 
        
    
    df.apply(_clean_unnecessary_spaces)
    if randomize:
        df = df.sample(frac=1).reset_index(drop=True)
    return df


if __name__ == "__main__":
    print(get_combined_dataset(True, True, True).shape)
