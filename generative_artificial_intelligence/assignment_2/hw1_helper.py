from os.path import dirname, join

import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import (
    get_data_dir,
    load_colored_mnist_text,
    load_pickled_data,
    load_text_data,
    save_distribution_1d,
    save_distribution_2d,
    save_text_to_plot,
    save_timing_plot,
    save_training_plot,
    savefig,
    show_samples,
)


# Question 1
def q1_sample_data_1():
    count = 1000
    rand = np.random.RandomState(0)
    samples = 0.2 + 0.2 * rand.randn(count)
    data = np.digitize(samples, np.linspace(0.0, 1.0, 20))
    split = int(0.8 * len(data))
    train_data, test_data = data[:split], data[split:]
    return train_data, test_data


def q1_sample_data_2():
    count = 10000
    rand = np.random.RandomState(0)
    a = 0.4 + 0.05 * rand.randn(count)
    b = 0.5 + 0.10 * rand.randn(count)
    c = 0.7 + 0.02 * rand.randn(count)
    mask = np.random.randint(0, 3, size=count)
    samples = np.clip(a * (mask == 0) + b * (mask == 1) + c * (mask == 2), 0.0, 1.0)

    data = np.digitize(samples, np.linspace(0.0, 1.0, 100))
    split = int(0.8 * len(data))
    train_data, test_data = data[:split], data[split:]
    return train_data, test_data


def visualize_q1_data(dset_type):
    if dset_type == 1:
        train_data, test_data = q1_sample_data_1()
        d = 20
    elif dset_type == 2:
        train_data, test_data = q1_sample_data_2()
        d = 100
    else:
        raise Exception("Invalid dset_type:", dset_type)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title("Train Data")
    ax1.hist(train_data, bins=np.arange(d) - 0.5, density=True)
    ax1.set_xlabel("x")
    ax2.set_title("Test Data")
    ax2.hist(test_data, bins=np.arange(d) - 0.5, density=True)
    print(f"Dataset {dset_type}")
    plt.show()


def q1_save_results(dset_type, part, fn):
    if dset_type == 1:
        train_data, test_data = q1_sample_data_1()
        d = 20
    elif dset_type == 2:
        train_data, test_data = q1_sample_data_2()
        d = 100
    else:
        raise Exception("Invalid dset_type:", dset_type)

    train_losses, test_losses, distribution = fn(train_data, test_data, d, dset_type)
    assert np.allclose(
        np.sum(distribution), 1
    ), f"Distribution sums to {np.sum(distribution)} != 1"

    print(f"Final Test Loss: {test_losses[-1]:.4f}")

    save_training_plot(
        train_losses,
        test_losses,
        f"Q1({part}) Dataset {dset_type} Train Plot",
        f"results/q1_{part}_dset{dset_type}_train_plot.png",
    )
    save_distribution_1d(
        train_data,
        distribution,
        f"Q1({part}) Dataset {dset_type} Learned Distribution",
        f"results/q1_{part}_dset{dset_type}_learned_dist.png",
    )


# Question 2
def q2a_save_results(dset_type, q2_a):
    data_dir = get_data_dir(1)
    if dset_type == 1:
        train_data, test_data = load_pickled_data(join(data_dir, "shapes.pkl"))
        img_shape = (20, 20)
    elif dset_type == 2:
        train_data, test_data = load_pickled_data(join(data_dir, "mnist.pkl"))
        img_shape = (28, 28)
    else:
        raise Exception()

    results = q2_a(
        train_data, test_data, img_shape, dset_type
    )
    if len(results) == 4:
        train_losses, test_losses, samples, model = results
    else:
        train_losses, test_losses, samples = results
        model = None
        
    samples = samples.astype("float32") * 255

    print(f"Final Test Loss: {test_losses[-1]:.4f}")
    save_training_plot(
        train_losses,
        test_losses,
        f"Q2(a) Dataset {dset_type} Train Plot",
        f"results/q2_a_dset{dset_type}_train_plot.png",
    )
    show_samples(samples, f"results/q2_a_dset{dset_type}_samples.png")
    return model


def q2b_save_results(dset_type, model, q2_b):
    data_dir = get_data_dir(1)
    if dset_type == 1:
        _, test_data = load_pickled_data(join(data_dir, "shapes.pkl"))
    elif dset_type == 2:
        _, test_data = load_pickled_data(join(data_dir, "mnist.pkl"))
    else:
        raise Exception()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    noise_ratios = [0.1, 0.3, 0.5]
    N = 10

    true_nll, perturbed_nlls, test_samples, perturbed_samples_dict = q2_b(
        model, test_data, device, noise_ratios=noise_ratios, N=N
    )

    col_w = 12
    print("=" * (10 + 14 + len(noise_ratios) * 15))
    print("  Average NLL Comparison")
    print("=" * (10 + 14 + len(noise_ratios) * 15))
    print(f"\n{'Sample':>8} | {'True Data':>{col_w}}", end="")
    for r in noise_ratios:
        print(f" | {f'Flip {int(r*100)}%':>{col_w}}", end="")
    print()
    print("-" * (10 + 14 + len(noise_ratios) * 15))

    for i in range(N):
        print(f"  #{i+1:>4}   | {true_nll[i]:>{col_w}.4f}", end="")
        for r in noise_ratios:
            print(f" | {perturbed_nlls[r][i]:>{col_w}.4f}", end="")
        print()

    print()
    print(f"  {'Mean':>6}  | {true_nll.mean():>{col_w}.4f}", end="")
    for r in noise_ratios:
        print(f" | {perturbed_nlls[r].mean():>{col_w}.4f}", end="")
    print("\n")

    # --- NLL comparison plot ---
    labels = ["True Data"] + [f"Flip {int(r*100)}%" for r in noise_ratios]
    all_nlls = [true_nll] + [perturbed_nlls[r] for r in noise_ratios]
    n_conditions = len(labels)
    x = np.arange(N)
    bar_width = 0.8 / n_conditions
    colors = plt.cm.coolwarm(np.linspace(0.1, 0.9, n_conditions))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                   gridspec_kw={"width_ratios": [3, 1]})

    # Left: per-sample grouped bar chart
    for i, (label, nll_vals, color) in enumerate(zip(labels, all_nlls, colors)):
        offset = (i - n_conditions / 2 + 0.5) * bar_width
        ax1.bar(x + offset, nll_vals, width=bar_width, label=label, color=color, alpha=0.85)

    ax1.set_xlabel("Sample index")
    ax1.set_ylabel("Average NLL (lower = higher likelihood)")
    ax1.set_title(f"Q2(b) Dataset {dset_type} — Per-sample Average NLL")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"#{i+1}" for i in range(N)])
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # Right: mean NLL per condition
    means = [nll.mean() for nll in all_nlls]
    bars = ax2.bar(labels, means, color=colors, alpha=0.85)
    ax2.set_ylabel("Mean Average NLL")
    ax2.set_title("Mean NLL by Condition")
    ax2.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for bar, val in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    save_path = f"results/q2_b_dset{dset_type}_nll_comparison.png"
    plt.savefig(save_path, bbox_inches="tight", dpi=100)
    plt.show()
    print(f"Saved: {save_path}")

    # --- Image grid: True Data vs. Bit-Flipped Perturbations ---
    row_labels = ["True Data"] + [f"Flip {int(r*100)}%" for r in noise_ratios]
    plot_data_list = [torch.from_numpy(test_samples)] + [
        perturbed_samples_dict[r] for r in noise_ratios
    ]
    n_rows = len(row_labels)

    fig2, axes = plt.subplots(n_rows, N, figsize=(N * 1.5, n_rows * 1.5))
    for row, (label, data) in enumerate(zip(row_labels, plot_data_list)):
        for col in range(N):
            img = (
                data[col].squeeze().numpy()
                if hasattr(data[col], "numpy")
                else data[col].squeeze()
            )
            axes[row, col].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(label, fontsize=9, rotation=90, labelpad=4)

    plt.suptitle(
        f"Q2(b) Dataset {dset_type} — True Data vs. Bit-Flipped Perturbations",
        fontsize=12,
        y=1.01,
    )
    plt.tight_layout()
    save_path2 = f"results/q2_b_dset{dset_type}_samples.png"
    plt.savefig(save_path2, bbox_inches="tight", dpi=100)
    plt.show()
    print(f"Saved: {save_path2}")



def visualize_q2a_data(dset_type):
    data_dir = get_data_dir(1)
    if dset_type == 1:
        train_data, test_data = load_pickled_data(join(data_dir, "shapes.pkl"))
        name = "Shape"
    elif dset_type == 2:
        train_data, test_data = load_pickled_data(join(data_dir, "mnist.pkl"))
        name = "MNIST"
    else:
        raise Exception("Invalid dset type:", dset_type)

    idxs = np.random.choice(len(train_data), replace=False, size=(100,))
    images = train_data[idxs].astype("float32") / 1 * 255
    show_samples(images, title=f"{name} Samples")

