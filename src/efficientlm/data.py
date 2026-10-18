"""Data loading. Real runs use SST-2 (GLUE); --smoke runs use a synthetic task so the
whole pipeline can be verified offline in about a minute."""
import random
from functools import lru_cache

import torch
from torch.utils.data import DataLoader


# ----------------------------------------------------------------------- SST-2
@lru_cache(maxsize=4)
def load_sst2(max_len, train_size, dev_size, data_seed, tokenizer_name):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    ds = load_dataset("glue", "sst2")

    # train -> (dev, train) carved from the official train split; test = official validation split.
    # The split depends only on data_seed, so every method/seed sees exactly the same data.
    full = ds["train"].shuffle(seed=data_seed)
    dev = full.select(range(dev_size))
    rest = full.select(range(dev_size, len(full)))
    if train_size and train_size < len(rest):
        rest = rest.select(range(train_size))
    test = ds["validation"]

    def encode(batch):
        return tok(batch["sentence"], truncation=True, max_length=max_len, return_token_type_ids=False)

    cols = ["input_ids", "attention_mask", "label"]
    out = []
    for split in (rest, dev, test):
        split = split.map(encode, batched=True)
        split.set_format(type=None, columns=cols)
        out.append(split)
    return tuple(out) + (tok,)


def make_collate(tokenizer):
    def collate(features):
        labels = torch.tensor([f["label"] for f in features], dtype=torch.long)
        batch = tokenizer.pad(
            [{"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]} for f in features],
            return_tensors="pt",
        )
        batch = dict(batch)
        batch["labels"] = labels
        return batch

    return collate


# ------------------------------------------------------------------- synthetic
def make_synthetic(vocab_size=100, seq_len=16, n_train=512, n_dev=128, n_test=128, seed=0):
    """Binary task: label = 1 iff the mean token id is above vocab_size/2. Learnable, tiny."""
    rng = random.Random(seed)

    def build(n):
        items = []
        for _ in range(n):
            ids = [rng.randrange(1, vocab_size) for _ in range(seq_len)]
            label = int(sum(ids) / seq_len > vocab_size / 2)
            items.append({"input_ids": ids, "attention_mask": [1] * seq_len, "label": label})
        return items

    return build(n_train), build(n_dev), build(n_test)


def stack_collate(features):
    return {
        "input_ids": torch.tensor([f["input_ids"] for f in features], dtype=torch.long),
        "attention_mask": torch.tensor([f["attention_mask"] for f in features], dtype=torch.long),
        "labels": torch.tensor([f["label"] for f in features], dtype=torch.long),
    }


# --------------------------------------------------------------------- loaders
def make_loaders(train_ds, dev_ds, test_ds, collate_fn, batch_size, seed):
    g = torch.Generator()
    g.manual_seed(seed)  # shuffling order depends on the run seed, not the data seed
    train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, generator=g)
    dev = DataLoader(dev_ds, batch_size=batch_size * 2, shuffle=False, collate_fn=collate_fn)
    test = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False, collate_fn=collate_fn)
    return train, dev, test
