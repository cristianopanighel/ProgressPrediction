import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import subprocess
import torch

from dotenv import load_dotenv
from torchvision import transforms
from torchvision.io import read_image
from tqdm import tqdm

load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--root', type=str, default=os.environ.get("MAIN"))
    parser.add_argument('--save_dir', type=str, default='bars')  # ucf24

    parser.add_argument('--num_videos', type=int, default=1000)
    parser.add_argument('--actions_per_video', type=int, default=4)
    parser.add_argument('--notches_per_action', type=int, default=4)

    parser.add_argument('--min_frames_per_notch', type=int, default=1)
    parser.add_argument('--max_frames_per_notch', type=int, default=1)
    parser.add_argument('--min_video_multiplier', type=int, default=1)
    parser.add_argument('--max_video_multiplier', type=int, default=1)

    parser.add_argument('--notch_width', type=int, default=2)
    parser.add_argument('--notch_height', type=int, default=32)
    parser.add_argument('--video_size', type=int, default=32)

    parser.add_argument('--visualise', action='store_true')

    return parser.parse_args()


COLOURS = torch.Tensor([(251, 75, 75), (255, 193, 99),
                       (254, 255, 92), (192, 255, 51)]) / 255


def create_frame(notch_list, notch_width: int, notch_height: int, size: int) -> torch.Tensor:
    frame = torch.zeros(3, size, size)
    for i, colour in enumerate(notch_list):
        frame[:, 0:notch_height, i *
              notch_width:(i+1)*notch_width] = colour[:, None, None]
    return frame


def create_data(args):
    save_dir = os.path.join(args.root, args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'rgb-images'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'splitfiles'), exist_ok=True)
    transform = transforms.ToPILImage()

    for video_index in tqdm(range(args.num_videos)):
        video_dir = os.path.join(save_dir, 'rgb-images', f'{video_index:05d}')
        os.makedirs(video_dir, exist_ok=True)
        if args.min_video_multiplier == args.max_video_multiplier:
            video_multiplier = args.min_video_multiplier
        else:
            video_multiplier = random.randint(
                args.min_video_multiplier, args.max_video_multiplier)
        notch_list = []
        frame_index = 0
        for action_index in range(args.actions_per_video):
            for notch_index in range(args.notches_per_action):
                if args.min_frames_per_notch == args.max_frames_per_notch:
                    notch_length = args.min_frames_per_notch * video_multiplier
                else:
                    notch_length = random.randint(
                        args.min_frames_per_notch, args.max_frames_per_notch) * video_multiplier
                notch_list.append(COLOURS[action_index])
                frame_array = create_frame(
                    notch_list, args.notch_width, args.notch_height, args.video_size)
                frame = transform(frame_array)

                for _ in range(notch_length):
                    frame_path = os.path.join(
                        video_dir, f'{frame_index:05}.jpg')
                    frame.save(frame_path)
                    frame_index += 1

    video_names = [f'{video_id:05d}\n' for video_id in range(args.num_videos)]
    random.shuffle(video_names)
    # with open(os.path.join(save_dir, 'splitfiles', 'train.txt'), 'w+') as f:
    #     f.writelines(sorted(video_names[:int(0.9 * args.num_videos)]))
    # with open(os.path.join(save_dir, 'splitfiles', 'test.txt'), 'w+') as f:
    #     f.writelines(sorted(video_names[int(0.9 * args.num_videos):]))

    with open(os.path.join(save_dir, 'splitfiles/', 'test.npy'), 'wb') as f:
        np.save(f, sorted(video_names[int(0.9 * args.num_videos):]))
    with open(os.path.join(save_dir, 'splitfiles/', 'train.npy'), 'wb') as f:
        np.save(f, sorted(video_names[:int(0.9 * args.num_videos)]))


def visualise(args):
    data_dir = os.path.join(args.root, args.save_dir, 'rgb-images')
    video_names = sorted(os.listdir(data_dir))
    os.makedirs("./plots/bars/", exist_ok=True)
    # os.makedirs("./plots/ucf24/", exist_ok=True)
    # video_names = [video_names[0], video_names[4], video_names[5], video_names[45]]
    video_names = [video_names[16]]

    frames_per_video = {}
    max_num_frames = 0
    for video_name in video_names:
        if '.DS_Store' in video_name:
            continue
        else:
            video_path = os.path.join(data_dir, video_name)
            num_frames = len(os.listdir(video_path))

        frames_per_video[video_name] = num_frames
        max_num_frames = max(num_frames, max_num_frames)

    for i in range(max_num_frames):
        frames = []
        progress = []
        for video_name in video_names:
            # only for ucf24
            # i = i+1
            frame_index = min(i, frames_per_video[video_name] - 1)
            frame_path = os.path.join(
                data_dir, video_name, f'{frame_index:05d}.jpg')
            print("frame path: " + str(frame_path))
            print("index: " + str(i))
            frames.append(read_image(frame_path))
            progress.append((i + 1) / frames_per_video[video_name])
        fig, axs = plt.subplots(1, 1, figsize=(2.5, 2.5))
        for frame, prog, ax in zip(frames, progress, [axs]):
            prog = min(1, prog)
            ax.imshow(transforms.ToPILImage()(frame))
            ax.axis('off')
            ax.set_title(f'{(prog * 100):.1f}%', y=-0.25, x=0.5, fontsize=10)
            # ax.set_axis('off')

        # plt.figure(figsize=(8, 8))
        # plt.imshow(np.transpose(grid, [1, 2, 0]))
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'./plots/bars/{i:03d}.jpg')
        plt.savefig(f'./plots/bars/{i:03d}.pdf')
        # only for ucf24
        # plt.savefig(f'./plots/ucf24/{i:03d}.jpg')
        # plt.savefig(f'./plots/ucf24/{i:03d}.pdf')

        plt.clf()

    subprocess.call([
        'ffmpeg', '-framerate', '6', '-i', './plots/bars/%03d.jpg', '-r', '30', '-pix_fmt', 'yuv420p',
        './plots/bars/out.mp4'
    ])
    # only for ucf24
    # subprocess.call([
    #    'ffmpeg', '-framerate', '6', '-i', './plots/ucf24/%03d.jpg', '-r', '30', '-pix_fmt', 'yuv420p',
    #    './plots/ucf24/out.mp4'
    # ])


def main():
    args = parse_args()
    random.seed(args.seed)
    # if args.visualise:
    visualise(args)
    # else:
    #    create_data(args)


if __name__ == '__main__':
    main()
