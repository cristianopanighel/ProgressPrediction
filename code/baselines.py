import os
import torch
import torch.nn as nn

from datasets import ImageDataset, UCFDataset
from dotenv import load_dotenv

load_dotenv()

DATA_ROOT = os.environ.get('MAIN')

def calc_baseline(train_lengths, test_lengths):

    max_length = max(train_lengths)
    loss = nn.L1Loss(reduction="sum")
    averages = torch.zeros(max(train_lengths))
    counts = torch.zeros(max(train_lengths))
    for length in train_lengths:
        progress = torch.arange(1, length + 1) / length
        averages[:length] += progress
        counts[:length] += 1
    averages = averages / counts

    count = 0
    average_loss = 0.0
    mid_loss = 0.5
    random_loss = 0.5
    for length in test_lengths:
        l = min(length, max_length)
        progress = torch.arange(1, length + 1) / length

        average_predictions = torch.ones(length)
        average_predictions[:l] = averages[:l]
        average_loss += loss(average_predictions * 100, progress * 100).item()
        mid_loss += loss(torch.full_like(progress, 0.5)
                         * 100, progress * 100).item()
        random_loss += loss(torch.rand_like(progress) *
                            100, progress * 100).item()

        count += length

    length = max(max(test_lengths), max_length)
    predictions = torch.ones(length)
    predictions[:max_length] = averages[:max_length]

    return "average_loss " + str(average_loss / count), "mid_loss " + str(mid_loss / count), "random_loss " + str(random_loss / count)


def ucf_baseline():
    trainset = UCFDataset(
        os.path.join(DATA_ROOT, "ucf24"),
        "rgb-images",
        "train.txt",
        True,
        False,
        1,
        False,
        False,
        1,
        "none",
        1,
    )
    testset = UCFDataset(
        os.path.join(DATA_ROOT, "ucf24"),
        "rgb-images",
        "test.txt",
        True,
        False,
        1,
        False,
        False,
        1,
        "none",
        1,
    )
    losses = calc_baseline(trainset.lengths, testset.lengths)
    print(f"--- ucf ---")
    print("average", losses[0])
    print("0.5", losses[1])
    print("random", losses[2])


def bf_baseline():
    losses = [0, 0, 0]
    for i in range(1, 5):
        trainset = ImageDataset(
            os.path.join(DATA_ROOT, "breakfast"),
            "rgb-images",
            f"train_s{i}.txt",
            False,
            1,
            False,
            False,
            1,
            False
        )
        testset = ImageDataset(
            os.path.join(DATA_ROOT, "breakfast"),
            "rgb-images",
            f"test_s{i}.txt",
            False,
            1,
            False,
            False,
            1,
            False
        )
    for i, loss in enumerate(calc_baseline(trainset.lengths, testset.lengths)):
        losses[i] += float(loss.split()[1]) / 4
    print(f"--- bf all ---")
    print("average", losses[0])
    print("0.5", losses[1])
    print("random", losses[2])

    # Change value of subsample_fps
    losses = [0, 0, 0]
    for i in range(1, 5):
        trainset = ImageDataset(
            os.path.join(DATA_ROOT, "breakfast"),
            "rgb-images",
            f"train_s{i}.txt",
            False,
            15,
            False,
            False,
            1,
            False
        )
        testset = ImageDataset(
            os.path.join(DATA_ROOT, "breakfast"),
            "rgb-images",
            f"test_s{i}.txt",
            False,
            15,
            False,
            False,
            1,
            False
        )
    for i, loss in enumerate(calc_baseline(trainset.lengths, testset.lengths)):
        losses[i] += float(loss.split()[1]) / 4
    print(f"--- bf all (sampled) ---")
    print("average", losses[0])
    print("0.5", losses[1])
    print("random", losses[2])


def bars_baseline():
    train = ImageDataset(os.path.join(DATA_ROOT, "bars"), "rgb-images",
                         f"train.txt", False, 1, False, False, 1, False)
    test = ImageDataset(os.path.join(DATA_ROOT, "bars"), "rgb-images",
                        f"test.txt", False, 1, False, False, 1, False)
    losses = calc_baseline(train.lengths, test.lengths)
    print(f"--- bars ---")
    print("average", losses[0])
    print("0.5", losses[1])
    print("random", losses[2])


def main():
    bars_baseline()
    ucf_baseline()
    bf_baseline()


if __name__ == "__main__":
    main()
