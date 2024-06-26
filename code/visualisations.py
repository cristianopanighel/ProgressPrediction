import argparse
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import string
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from datasets import ImageDataset, FeatureDataset, UCFDataset, Middle
from dotenv import load_dotenv
from PIL import Image
from typing import Dict, List

load_dotenv()

# Constants
DATA_ROOT = os.environ.get("MAIN")
BARS_IMAGES = os.environ.get("BARS_IMAGES")
BARS = os.environ.get("BARS")

UCF_IMAGES = os.environ.get("UCF_IMAGES")
UCF = os.environ.get("UCF")
UCF_RESNET18 = os.environ.get("UCF_RESNET18")
UCF_SEGMENTS = os.environ.get("UCF_SEGMENTS")
UCF_VGG_SEGMENTS = os.environ.get("UCF_VGG_SEGMENTS")
UCF_RESNET18_SEGMENTS = os.environ.get("UCF_RESNET18_SEGMENTS")
UCF_RESNET18_MASK_SEGMENTS = os.environ.get("UCF_RESNET18_MASK_SEGMENTS")
UCF_MASK = os.environ.get("UCF_MASK")
UCF_MASK_RANDOM_PE = os.environ.get("UCF_MASK_RANDOM_PE")
UCF_MASK_REVERSE_PE = os.environ.get("UCF_MASK_REVERSE_PE")
UCF_TF = os.environ.get("UCF_TF")

UCF_NUOVO_SEGMENTS = os.environ.get("UCF_NUOVO_SEGMENTS")
UCF_NUOVO_MASK_SEGMENTS = os.environ.get("UCF_NUOVO_MASK_SEGMENTS")

BREAKFAST_IMAGES = os.environ.get("BREAKFAST_IMAGES")
BREAKFAST = os.environ.get("BREAKFAST")
BREAKFAST_RESNET18 = os.environ.get("BREAKFAST_RESNET18")
BREAKFAST_MASK = os.environ.get("BREAKFAST_MASK")
BREAKFAST_SEGMENTS = os.environ.get("BREAKFAST_SEGMENTS")
BREAKFAST_RESNET18_SEGMENTS = os.environ.get("BREAKFAST_RESNET18_SEGMENTS")
BREAKFAST_RESNET18_MASK_SEGMENTS = os.environ.get("BREAKFAST_RESNET18_MASK_SEGMENTS")

BAR_WIDTH = 0.5
SPACING = 1.5
MODE_COLOURS = {
    "'full-video' inputs": (0.2980392156862745, 0.4470588235294118, 0.6901960784313725),
    "'random-noise' inputs": (0.8666666666666667, 0.5176470588235295, 0.3215686274509804),
    "'video-segments' inputs": (0.3333333333333333, 0.6588235294117647, 0.40784313725490196),
    "'frame-indices' inputs": (0.7686274509803922, 0.3058823529411765, 0.3215686274509804),
    "MAE": (0.3333333333333333, 0.6588235294117647, 0.40784313725490196),
    "MSE": (0.7686274509803922, 0.3058823529411765, 0.3215686274509804),
}
BASELINES = ["average-index", "static-0.5", "random"]
LINEWIDTH = 3
TITLE_X_OFFSET = 0.5
TITLE_Y_OFFSET = -0.25

# Matplotlib parameters
plt.style.use("seaborn-v0_8-paper")
plt.rcParams['axes.axisbelow'] = True


def set_font_sizes(small=12, medium=14, big=16):
    plt.rc('font', size=small)          # controls default text sizes
    plt.rc('axes', titlesize=small)     # fontsize of the axes title
    plt.rc('axes', labelsize=medium)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=small)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=small)    # fontsize of the tick labels
    plt.rc('legend', fontsize=small)    # legend fontsize
    plt.rc('figure', titlesize=big)  # fontsize of the figure title


def set_spines(enable: bool):
    plt.rcParams['axes.spines.left'] = enable
    plt.rcParams['axes.spines.right'] = enable
    plt.rcParams['axes.spines.top'] = enable
    plt.rcParams['axes.spines.bottom'] = enable


# Datasets
if not os.path.isfile('./data/lengths.json'):
    os.makedirs('./data/', exist_ok=True)
    print('loading ucf101')
    ucf101 = {
        'all': sorted(UCFDataset(os.path.join(DATA_ROOT, "ucf24"), "rgb-images", f"all.txt").lengths),
        'train': [sorted(UCFDataset(os.path.join(DATA_ROOT, "ucf24"), "rgb-images", f"train.txt").lengths)],
        'test': [sorted(UCFDataset(os.path.join(DATA_ROOT, "ucf24"), "rgb-images", f"test.txt").lengths)]
    }
    print('loading breakfast')
    breakfast = {
       'all': sorted(ImageDataset(os.path.join(DATA_ROOT, "breakfast"), "rgb-images", f"all.txt").lengths),
       'train': [sorted(ImageDataset(os.path.join(DATA_ROOT, "breakfast"), "rgb-images", f"train_s{i}.txt").lengths) for i in range(1, 5)],
       'test': [sorted(ImageDataset(os.path.join(DATA_ROOT, "breakfast"), "rgb-images", f"test_s{i}.txt").lengths) for i in range(1, 5)],
    }
    print('loading bars')
    bars = {
        'all': sorted(ImageDataset(os.path.join(DATA_ROOT, "bars"), "rgb-images", f"all.txt").lengths),
        'train': [sorted(ImageDataset(os.path.join(DATA_ROOT, "bars"), "rgb-images", f"train.txt").lengths)],
        'test': [sorted(ImageDataset(os.path.join(DATA_ROOT, "bars"), "rgb-images", f"test.txt").lengths)]
    }
    with open('./data/lengths.json', 'w+') as f:
        json.dump({
            'ucf101': ucf101,
            'breakfast': breakfast,
            'bars': bars
        }, f)
else:
    with open('./data/lengths.json') as f:
        data = json.load(f)
    ucf101 = data['ucf101']
    breakfast = data['breakfast']
    bars = data['bars']

# Helper functions


def load_results(path: str):
    with open(path) as f:
        results = json.load(f)
    return results


def calc_baselines(train_lengths, test_lengths):
    max_length = max(train_lengths)
    loss = nn.L1Loss(reduction="sum")
    averages = torch.zeros(max(train_lengths))
    counts = torch.zeros(max(train_lengths))
    for length in train_lengths:
        progress = torch.arange(1, length + 1) / length
        averages[:length] += progress
        counts[:length] += 1
    averages = averages / counts

    index_loss, static_loss, random_loss, count = 0, 0, 0, 0
    for length in test_lengths:
        l = min(length, max_length)
        progress = torch.arange(1, length + 1) / length

        average_predictions = torch.ones(length)
        average_predictions[:l] = averages[:l]
        index_loss += loss(average_predictions * 100, progress * 100).item()
        static_loss += loss(torch.full_like(progress, 0.5)
                            * 100, progress * 100).item()
        random_loss += loss(torch.rand_like(progress) *
                            100, progress * 100).item()

        count += length

    length = max(max(test_lengths), max_length)
    predictions = torch.ones(length)
    predictions[:max_length] = averages[:max_length]

    return predictions, index_loss / count, static_loss / count, random_loss / count


def calculate_average_baseline(trains, tests):
    num_sets = len(trains)
    max_length = 0
    for (train, test) in zip(trains, tests):
        max_length = max(max_length, max(train), max(test))

    average_predictions = torch.zeros(max_length)
    avg_index_loss, avg_static_loss, avg_random_loss = 0, 0, 0
    for (train, test) in zip(trains, tests):
        predictions, index_loss, static_loss, random_loss = calc_baselines(
            train, test)
        avg_index_loss += index_loss / num_sets
        avg_static_loss += static_loss / num_sets
        avg_random_loss += random_loss / num_sets
        average_predictions += predictions / num_sets

    return average_predictions, avg_index_loss, avg_static_loss, avg_random_loss

# Plots

def plot_errors_class(results: Dict, dataset: str, modes: List[str], names: List[str]):
    set_spines(False)
    plt.figure(figsize=(7.2, 5.2))

    data = [[] for _ in modes]
    xs_indices, networks = zip(
        *[(i, key) for i, key in enumerate(results[dataset]) if key not in BASELINES])
    xs = np.array(list(xs_indices))
    n = np.array(list(networks))

    for network in n:
        for i, mode in enumerate(modes):
            if mode in results[dataset][network]:
                data[i].append(results[dataset][network][mode])
            else:
                data[i].append(0)
    
    for i, (values, mode) in enumerate(zip(data, modes)):
        bar_xs = xs * SPACING + i * BAR_WIDTH
        print(bar_xs, values, mode, dataset)
        plt.bar(bar_xs, values, width=BAR_WIDTH,
                label=mode, color=MODE_COLOURS[mode])
    # xticks = xs_indices * SPACING + BAR_WIDTH * 0.5
    xticks = xs * SPACING + BAR_WIDTH * 0.5
    plt.grid(axis='y')
    plt.axhline(y=0, linestyle='-', color='grey', zorder=-1)
    plt.xticks(rotation=90)
    plt.xticks(xticks, n)
    yticks = [0, 5, 10, 15, 20, 25]
    plt.tick_params(axis='y', length=0)
    plt.yticks(yticks, [f'{tick}%' for tick in yticks])
    plt.ylabel("Error")
    plt.legend(loc='upper left', fancybox=False, shadow=False, ncol=3, fontsize = 'x-small')
    plt.tight_layout()
    filename = f"./plots/results/{dataset}_{'_'.join(names).replace(' ', '_')}"
    plt.show()
    plt.savefig(f"{filename}.{FILE}")
    plt.clf()
    set_spines(True)

def plot_result_bar(results: Dict, dataset: str, modes: List[str], names: List[str]):
    set_spines(False)
    plt.figure(figsize=(7.2, 5.2))

    data = [[] for _ in modes]
    xs_indices, networks = zip(
        *[(i, key) for i, key in enumerate(results[dataset]) if key not in BASELINES])
    xs = np.array(list(xs_indices))
    n = np.array(list(networks))

    for network in n:
        for i, mode in enumerate(modes):
            if mode in results[dataset][network]:
                data[i].append(results[dataset][network][mode])
            else:
                data[i].append(0)
    
    for i, (values, mode) in enumerate(zip(data, modes)):
        bar_xs = xs * SPACING + i * BAR_WIDTH
        print(bar_xs, values, mode, dataset)
        plt.bar(bar_xs, values, width=BAR_WIDTH,
                label=mode, color=MODE_COLOURS[mode])
    # xticks = xs_indices * SPACING + BAR_WIDTH * 0.5
    xticks = xs * SPACING + BAR_WIDTH * 0.5
    if dataset != 'Bars':
        plt.axhline(
            y=results[dataset]["average-index"]["'full-video' inputs"],
            xmax=0.5,
            linestyle="-",
            label="average-index",
            color=(0.8, 0.7254901960784313, 0.4549019607843137),
            linewidth=LINEWIDTH,
        )
        plt.axhline(
            y=results[dataset]["static-0.5"]["'full-video' inputs"],
            xmax=0.5,
            linestyle="-",
            label="static-0.5",
            color=(0.8549019607843137, 0.5450980392156862, 0.7647058823529411),
            linewidth=LINEWIDTH,
        )
        plt.axhline(
            y=results[dataset]["random"]["'full-video' inputs"],
            xmax=0.5,
            linestyle="-",
            label="random",
            color=(0.5058823529411764, 0.4470588235294118, 0.7019607843137254),
            linewidth=LINEWIDTH,
        )

    plt.grid(axis='y')
    plt.axhline(y=0, linestyle='-', color='grey', zorder=-1)
    plt.xticks(xticks, n)
    if dataset == 'Bars':
        yticks = [0, 0.5, 1, 1.5]
    else:
        yticks = [0, 5, 10, 15, 20, 25, 30, 35]
    plt.tick_params(axis='y', length=0)
    plt.yticks(yticks, [f'{tick}%' for tick in yticks])
    plt.ylabel("Error")
    plt.legend(loc='upper center', bbox_to_anchor=(
        0.5, -0.15), fancybox=False, shadow=False, ncol=3, fontsize = 'x-small')
    plt.tight_layout()
    filename = f"./plots/results/{dataset}_{'_'.join(names).replace(' ', '_')}"
    plt.savefig(f"{filename}.{FILE}")
    plt.clf()
    set_spines(True)

def plot_baselines():
    bf_predictions, bf_index_loss, bf_static_loss, bf_random_loss = calculate_average_baseline(
        breakfast['train'], breakfast['test'])
    ucf_predictions, ucf_index_loss, ucf_static_loss, ucf_random_loss = calculate_average_baseline(
        ucf101['train'], ucf101['test'])
    bars_predictions, bars_index_loss, bars_static_loss, bars_random_loss = calculate_average_baseline(
        bars['train'], bars['test'])
    print('bf', bf_index_loss, bf_static_loss, bf_random_loss)
    print('ucf', ucf_index_loss, ucf_static_loss, ucf_random_loss)
    print('bars', bars_index_loss, bars_static_loss, bars_random_loss)

    figure, axs = plt.subplots(1, 3, figsize=(19.2, 4.8 * 1.3))
    axs[0].plot(bf_predictions, label="average-index")
    axs[1].plot(ucf_predictions, label="average-index")
    axs[2].plot(bars_predictions, label="average-index")

    with open('./data/bf_baseline.txt', 'w+') as f:
        f.write('\n'.join([str(val) for val in bf_predictions.tolist()]))
    with open('./data/ucf_baseline.txt', 'w+') as f:
        f.write('\n'.join([str(val) for val in ucf_predictions.tolist()]))
    with open('./data/bars_baseline.txt', 'w+') as f:
        f.write('\n'.join([str(val) for val in bars_predictions.tolist()]))

    for ax in axs.flat:
        ax.set_xlabel("Frame")
        ax.set_ylabel("Progress")
        ax.legend()
    axs[0].set_title(
       "(a) Average-index baseline on breakfast",
       y=TITLE_Y_OFFSET / 1.2,
       x=TITLE_X_OFFSET,
    )
    axs[1].set_title(
        "(b) Average-index baseline on UCF101-24",
        y=TITLE_Y_OFFSET / 1.2,
        x=TITLE_X_OFFSET,
    )
    axs[2].set_title(
        "(c) Average-index baseline on bars",
       y=TITLE_Y_OFFSET / 1.2,
       x=TITLE_X_OFFSET,
    )

    plt.tight_layout()
    plt.savefig(f"./plots/avg_index_baseline.{FILE}")
    plt.clf()

def plot_baseline_example():
    predictions, _, _, _ = calc_baselines([10, 20, 30], [50])

    plt.figure(figsize=(6.4*1.5, 4.8*1.5))
    plt.plot([random.random() for _ in predictions],
             label='random', linestyle=':', linewidth=3)
    plt.plot([0.5 for _ in predictions], label='static-0.5', linewidth=3)
    plt.plot(predictions, label="average-index", linewidth=3)
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], [
               '0%', '20%', '40%', '60%', '80%', '100%'])
    plt.xlabel('Frame')
    plt.ylabel('Progress')
    plt.xlim(0, 50)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"./plots/avg_index_example.{FILE}")
    plt.clf()

def make_length_plot(lengths, ax: plt.Axes, title: str, bucket_size: int = 10):
    buckets = {}
    mean = np.percentile(lengths, 50)
    for length in lengths:
        length = math.floor(length / bucket_size) * bucket_size
        if length not in buckets:
            buckets[length] = 0
        buckets[length] += 1
    ax.bar(buckets.keys(), buckets.values(), width=bucket_size)
    ax.axvline(mean, color='red', linestyle = ':')
    ax.set_title(title, y=TITLE_Y_OFFSET*1.1, x=TITLE_X_OFFSET)

def plot_dataset_lengths():
    figure, axs = plt.subplots(1, 3, figsize=(6.4 * 2, 4.4))
    make_length_plot(
        breakfast['all'], axs[0], '(a) video length distribution for breakfast', bucket_size=100)
    make_length_plot(
        ucf101['all'], axs[1], '(b) video length distribution for UCF101-24', bucket_size=10)
    make_length_plot(
        bars['all'], axs[2], '(c) video length distribution for synthetic dataset', bucket_size=10)
    for ax in axs.flat:
        ax.set_xlabel("Video Length")
        ax.set_ylabel("Number of Videos")
    plt.tight_layout()
    plt.savefig(f'./plots/dataset_lengths.{FILE}')
    plt.clf()

def plot_synthetic(video_index: int, frame_indices: List[int]):
    data_dir = os.environ.get("BARS_IMAGES")
    video_name = sorted(os.listdir(data_dir))[video_index]
    video_path = os.path.join(data_dir, video_name)
    frame_names = sorted(os.listdir(video_path))
    num_frames = len(frame_names)

    figure, axs = plt.subplots(1, len(frame_indices), figsize=(6.4, 2.4))
    for letter, ax, frame_index in zip(string.ascii_lowercase, axs, frame_indices):
        frame_name = frame_names[frame_index]
        frame_path = os.path.join(video_path, frame_name)
        frame = Image.open(frame_path)
        ax.imshow(frame)
        ax.axis('off')
        progress = (frame_index + 1) / num_frames
        ax.set_title(f'({letter}) \nt={frame_index}\np={round(progress * 100, 1)}%',
                     y=TITLE_Y_OFFSET * 2.5, x=TITLE_X_OFFSET)

    plt.tight_layout()
    plt.savefig(f'./plots/bars.{FILE}')
    plt.clf()

def visualise_video(video_dir: str, timestamps: List[int], result_paths: List[str], video_name: str, N: int, offset: int = 0, subsample: int = 1):
    gs = plt.GridSpec(len(timestamps), 4)
    fig = plt.figure(figsize=(6.4*2, 4.8*1.5))
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[1, 0])
    ax3 = plt.subplot(gs[:, 1:])
    axs = [ax1, ax2, ax3]

    frames = sorted(os.listdir(video_dir))[::subsample]
    num_frames = len(frames)
    index = timestamps[0]
    for i, index in enumerate(timestamps):
        frame_path = os.path.join(video_dir, frames[index+offset])
        frame = Image.open(frame_path)
        axs[i].imshow(frame)
        axs[i].axis('off')
        axs[i].set_title(f't={index}', fontsize=20)

    ground_truth = [(i+1) / num_frames for i in range(num_frames)]
    static = [0.5 for _ in range(num_frames)]
    for index in timestamps:
        axs[-1].axvline(index, color='red', linestyle=':')
    for name, path, linestyle in result_paths:
        with open(path) as f:
            data = [float(row.strip()) for row in f.readlines()][:num_frames]
        if name != 'average-index':
            axs[-1].plot(np.convolve(data, np.ones(N)/N, mode='full'),
                         label=name, linestyle=linestyle, linewidth=LINEWIDTH)
        else:
            axs[-1].plot(data, label=name, linewidth=LINEWIDTH)

    axs[-1].plot(ground_truth, label='Ground Truth', linewidth=LINEWIDTH)
    axs[-1].plot(static, label='Static', linewidth=LINEWIDTH)
    axs[-1].set_xlabel('Frame')
    axs[-1].set_ylabel('Progress')
    axs[-1].tick_params(axis='both', which='major')
    axs[-1].tick_params(axis='both', which='minor')

    plt.grid(axis='y')
    yticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ytick_labels = ['0%', '20%', '40%', '60%', '80%', '100%']
    axs[-1].tick_params(axis='y', length=0)
    axs[-1].set_yticks(yticks, ytick_labels)
    axs[-1].set_xlim(0, num_frames)

    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(f'./plots/examples/{video_name}.{FILE}')
    plt.clf()

def stats(dataset: str, splitfiles: List[str], length=False):
    root = os.path.join(DATA_ROOT, dataset)
    for splitfile in splitfiles:
        with open(f'{os.path.join(root, f"splitfiles/{splitfile}.txt")}') as f:
            lines = f.readlines()
        counts_per_class = {}
        num_frames_per_class = {}
        total = 0
        for line in lines:
            line = line.strip()
            if length:
                frames_path = os.path.join(root, 'rgb-images', line)
                num_frames = len(os.listdir(frames_path))
            if "breakfast" in dataset:
                activity_class = line.split('_')[-1]
            else:
                activity_class = line.split('/')[0]
            if activity_class not in counts_per_class:
                counts_per_class[activity_class] = 0
            if activity_class not in num_frames_per_class:
                num_frames_per_class[activity_class] = 0
            if length:
                num_frames_per_class[activity_class] += num_frames
            counts_per_class[activity_class] += 1
            total += 1

        plt.bar(counts_per_class.keys(), [
                num_frames_per_class[key] / counts_per_class[key] for key in counts_per_class])
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f'./plots/{dataset}_{splitfile}.png')
        plt.clf()
        print(
            f'--- {total} videos in {dataset}/{splitfile} ({len(counts_per_class)} classes) ---')
        for activity_class in counts_per_class:
            # class: number of examples per class (average number of frames per class)
            print(
                f'{activity_class}: {counts_per_class[activity_class]} ({num_frames_per_class[activity_class] / counts_per_class[activity_class]})')

def tube_stats(splitfile: str):
    dataset = FeatureDataset(os.path.join(DATA_ROOT, 'ucf24'), 'features/vgg11embed', splitfile, False, 1, False, False, 1, 'none', 1)
    counts_per_class = {}
    num_frames_per_class = {}
    total = 0
    for (name, data, _) in dataset:
        length = data.shape[0]
        activity_class = name.split('/')[0]
        if activity_class not in counts_per_class:
            counts_per_class[activity_class] = 0
        if activity_class not in num_frames_per_class:
            num_frames_per_class[activity_class] = 0
        if length:
            num_frames_per_class[activity_class] += length
        counts_per_class[activity_class] += 1
        total += 1
        plt.bar(counts_per_class.keys(), [
                num_frames_per_class[key] / counts_per_class[key] for key in counts_per_class])
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f'./plots/ucf24__{splitfile}.png')
        plt.clf()
    print(
        f'--- {total} videos in ucf24/{splitfile} ({len(counts_per_class)} classes) ---')
    for activity_class in counts_per_class:
        print(
            f'{activity_class}: {counts_per_class[activity_class]} ({num_frames_per_class[activity_class] / counts_per_class[activity_class]})')

def dataset_statistics():
    stats('ucf24', ['all', 'train', 'test'], length=True)
    tube_stats('test_embed.txt')
    stats('breakfast', ['all'], length=True)

def dataset_visualisations():
    transform = transforms.Resize((240, 320))
    dataset = UCFDataset(os.path.join(DATA_ROOT, "ucf24"),
                         "rgb-images", f"small.txt", sample_transform=Middle())
    num_activities = len(dataset) // 5
    # plt.rcParams.update({'font.size' : 10})
    frames = []
    fig, axs = plt.subplots(nrows = num_activities, ncols = 5, layout = 'constrained' , figsize=(6.4, 4.8*(24/5)))
    unique_names = []
    for name, frame, _ in dataset:
        frames.append(transform(frame[0]))
        activity_name = name.split('/')[0]
        if activity_name not in unique_names:
            unique_names.append(activity_name)
    for (frame, ax) in zip(frames, axs.flat):
        ax.imshow(frame)
        ax.set_xticks([])
        ax.set_xticks([], minor=True)
        ax.set_yticks([])
        ax.set_yticks([], minor=True)
    for (unique_name, ax) in zip(unique_names, axs[:, 0]):
        ax.text(-600, 150, unique_name)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig('./plots/ucf_visualisations.png')
    plt.clf()
    plt.close()

def visualise_results():
    # for index, timestamp in zip(['00015'], [(0, 15)]):
    #     visualise_video(
    #         os.path.join(BARS_IMAGES,index), timestamp,
    #         [('ProgressNet', os.path.join(BARS,f'{index}.txt'), '-.')],
    #         f'bars_video{index}', 1
    #     )
    # for index, timestamp in zip(['GolfSwing/v_GolfSwing_g01_c03', 'GolfSwing/v_GolfSwing_g01_c02', 'Biking/v_Biking_g01_c02'], [(0,25), (0,45), (0, 80)]): #zip(['Biking/v_Biking_g01_c02', 'Fencing/v_Fencing_g01_c01', 'FloorGymnastics/v_FloorGymnastics_g01_c03', 'GolfSwing/v_GolfSwing_g01_c03', 'GolfSwing/v_GolfSwing_g01_c02', 'HorseRiding/v_HorseRiding_g01_c01'], [(0,80), (0,45), (0,60), (0,25), (0,45), (0,125)]):
    #    visualise_video(
    #        os.path.join(UCF_IMAGES,f'{index}'), timestamp,
    #        [#('ProgressNet (VGG11)', os.path.join(UCF,f'{index.replace("/", "_")}_0.txt'), '-.'),
    #         #('ProgressNet (ResNet18)', os.path.join(UCF_RESNET18,f'{index.replace("/", "_")}_0.txt'), '-.'),
    #         ('ProgressNet (VGG11)',os.path.join(UCF_VGG_SEGMENTS,f'{index.replace("/", "_")}_0.txt'), '-.'),
    #         ('ProgressNet (ResNet18)',os.path.join(UCF_RESNET18_SEGMENTS,f'{index.replace("/", "_")}_0.txt'), '-.'),
    #         ('ProgressNet (masked Transformer)',os.path.join(UCF_RESNET18_MASK_SEGMENTS,f'{index.replace("/", "_")}_0.txt'), '-.'),
    #         #('ProgressNet (masked Transformer)', os.path.join(UCF_MASK,f'{index.replace("/", "_")}_0.txt'), '-.'),
    #         #('ProgressNet (random pe)', os.path.join(UCF_MASK_RANDOM_PE,f'{index.replace("/", "_")}_0.txt'), '-.'),
    #         #('ProgressNet (reverse pe)', os.path.join(UCF_MASK_REVERSE_PE,f'{index.replace("/", "_")}_0.txt'), '-.'),
    #         #('ProgressNet (tf)', os.path.join(UCF_TF,f'{index.replace("/", "_")}_0.txt'), '-.'),
    #         #('average-index', f'./data/ucf_baseline.txt', '-')
    #         ],
    #        f'ucf_video_{index.replace("/", "_")}', 1
    #    )
    for index, timestamp in zip(['P07_webcam01_P07_sandwich'], [(0, 130)]):#, 'P12_cam01_P12_pancake'], [(0,2135), (0,2870)]):
        visualise_video(
            os.path.join(BREAKFAST_IMAGES, f'{index}'), timestamp,
            [('ProgressNet (VGG11)', os.path.join(BREAKFAST_SEGMENTS, f'{index.replace("/", "_")}.txt'), '-.'),
             ('ProgressNet (ResNet18)', os.path.join(BREAKFAST_RESNET18_SEGMENTS, f'{index.replace("/", "_")}.txt'), '-.'),
             ('ProgressNet (masked Transformer)', os.path.join(BREAKFAST_RESNET18_MASK_SEGMENTS, f'{index.replace("/", "_")}.txt'), '-.'),
             ('average-index', f'./data/bf_baseline.txt', '-')],
             f'bf_video_{index.replace("/", "_")}', 1, subsample = 15
        )

parser = argparse.ArgumentParser()
parser.add_argument('--pdf', action='store_true')
args = parser.parse_args()
FILE = 'pdf' if args.pdf else 'png'


def main():
    try:
        os.makedirs('./plots/', exist_ok=True)
        os.makedirs('./plots/bars/', exist_ok=True)
        os.makedirs('./plots/results/', exist_ok=True)
        os.makedirs('./plots/examples/', exist_ok=True)
    except:
        pass

    # set_font_sizes(16, 18, 20)
    # plot_baselines()
    # plot_baseline_example()

    # dataset_visualisations()
    # dataset_statistics()

    # result plots
    # results = load_results(os.environ.get("RESULTS"))
    # for dataset in ["UCF101-24"]:#, "breakfast"]:
    #     plot_result_bar(results, dataset, [
    #                     "'full-video' inputs", "'random-noise' inputs"], ['full video', 'random'])
    #     plot_result_bar(results, dataset, [
    #                     "'video-segments' inputs", "'frame-indices' inputs"], ['video segments', 'indices'])
    # plot_result_bar(results, "Bars", [
    #                 "'full-video' inputs", "'video-segments' inputs"], ['full video', 'video segments'])
    # # average index baseline
    # set_font_sizes()
    # # dataset statistics
    # plot_dataset_lengths()
    # # syntethic dataset example
    # plot_synthetic(4, [0, 3, 7, 11, 15])
    # # example progress predictions
    set_font_sizes(16, 18, 20)
    visualise_results()
    # errors = load_results(os.environ.get("ERRORS"))
    # plot_errors_class(errors, "UCF101-24", ['MAE', 'MSE'], ['MAE', 'MSE'])


if __name__ == '__main__':
    main()
