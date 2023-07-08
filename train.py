import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from model import Discriminator, Generator
from utils import (
    gradient_penalty,
    save_checkpoint,
    plot_to_tensorboard,
    load_checkpoint,
    generate_examples,
)

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import config
from math import log2

torch.backends.cudnn.benchmark = True


def get_loader(image_size):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(
                [0.5 for _ in range(config.CHANNELS_IMG)],
                [0.5 for _ in range(config.CHANNELS_IMG)],
            ),
        ]
    )

    batch_size = config.BATCH_SIZES[int(log2(image_size / 4))]
    dataset = datasets.ImageFolder(config.DATASET, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    return loader, dataset


def train_fn(
    critic,
    gen,
    loader,
    dataset,
    step,
    alpha,
    opt_critic,
    opt_gen,
    tensorboard_step,
    writer,
    scaler_gen,
    scaler_critic,
):
    loop = tqdm(loader, leave=True)

    for batch_idx, (real, _) in enumerate(loop):
        real = real.to(config.DEVICE)
        cur_batch_size = real.shape[0]

        # Train Critic: max E[critic(real)] - E[critic(fake)]
        # equivalent to minimizing the negative of that
        with torch.cuda.amp.autocast():
            noise = torch.randn(cur_batch_size, config.Z_DIM, 1, 1).to(config.DEVICE)
            fake = gen(noise, alpha, step)

            critic_real = critic(real, alpha, step)
            critic_fake = critic(fake.detach(), alpha, step)

            gp = gradient_penalty(critic, real, fake, alpha, step, device=config.DEVICE)

            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))
                + config.LAMBDA_GP * gp
                + (0.001 * torch.mean(critic_real**2))
            )

        opt_critic.zero_grad()
        scaler_critic.scale(loss_critic).backward()
        scaler_critic.step(opt_critic)
        scaler_critic.update()

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        with torch.cuda.amp.autocast():
            gen_fake = critic(fake, alpha, step)

            loss_gen = -torch.mean(gen_fake)

        opt_gen.zero_grad()
        scaler_gen.scale(loss_gen).backward()
        scaler_gen.step(opt_gen)
        scaler_gen.update()

        alpha += cur_batch_size / (config.PROGRESSIVE_EPOCHS[step] * len(dataset) * 0.5)
        alpha = min(alpha, 1)

        # Print losses occasionally and print to tensorboard
        if batch_idx % 500 == 0:
            with torch.no_grad():
                fixed_fakes = gen(config.FIXED_NOISE, alpha, step) * 0.5 + 0.5
            plot_to_tensorboard(
                writer,
                loss_critic.item(),
                loss_gen.item(),
                real.detach(),
                fixed_fakes.detach(),
                tensorboard_step,
            )
            tensorboard_step += 1

    return tensorboard_step, alpha


def main():
    gen = Generator(
        config.Z_DIM, in_channels=config.IN_CHANNELS, img_channels=config.CHANNELS_IMG
    ).to(config.DEVICE)
    critic = Discriminator(in_channels=config.IN_CHANNELS, img_channels=config.CHANNELS_IMG).to(
        config.DEVICE
    )

    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))
    opt_critic = optim.Adam(critic.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))

    # Float16 to speed up training
    scaler_gen = torch.cuda.amp.GradScaler()
    scaler_critic = torch.cuda.amp.GradScaler()

    writer = SummaryWriter(f"logs/gan")

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN,
            gen,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC,
            critic,
            opt_critic,
            config.LEARNING_RATE,
        )

    gen.train()
    critic.train()

    tensorboard_step = 0
    step = int(log2(config.START_TRAIN_AT_IMG_SIZE / 4))

    for num_epoch in config.PROGRESSIVE_EPOCHS[step:]:
        # Alpha is used for fade-in effect. After training, alpha should be 1 otherwise you will use only part of the last layer
        alpha = 1e-5
        loader, dataset = get_loader(4 * 2**step)
        print(f"Current image size: {4 * 2 ** step}")

        for epoch in range(num_epoch):
            print(f"Epoch [{epoch+1}/{num_epoch}]")
            tensorboard_step, alpha = train_fn(
                critic,
                gen,
                loader,
                dataset,
                step,
                alpha,
                opt_critic,
                opt_gen,
                tensorboard_step,
                writer,
                scaler_gen,
                scaler_critic,
            )

            if config.SAVE_MODEL:
                save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
                save_checkpoint(critic, opt_critic, filename=config.CHECKPOINT_CRITIC)

        step += 1  # Progress to the next image size


if __name__ == "__main__":
    main()
