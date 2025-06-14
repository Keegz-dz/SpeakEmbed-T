"""
Display and visualization utilities for progress bars, tables, timing, and plotting.
"""

import time
import numpy as np
import sys


def progbar(i, n, size=16):
    """
    Create a progress bar string for the current progress.
    Args:
        i (int): Current step.
        n (int): Total steps.
        size (int): Length of the progress bar.
    Returns:
        str: Progress bar string.
    """
    done = (i * size) // n
    bar = ''
    for i in range(size):
        bar += '█' if i <= done else '░'
    return bar


def stream(message):
    """
    Print a message to stdout, handling non-ASCII characters.
    Args:
        message (str): Message to print.
    """
    try:
        sys.stdout.write("\r{%s}" % message)
    except:
        # Remove non-ASCII characters from message
        message = ''.join(i for i in message if ord(i)<128)
        sys.stdout.write("\r{%s}" % message)


def simple_table(item_tuples):
    """
    Print a simple table with headings and values.
    Args:
        item_tuples (list of tuples): Each tuple is (heading, value).
    """
    border_pattern = '+---------------------------------------'
    whitespace = '                                            '

    headings, cells, = [], []

    for item in item_tuples:
        heading, cell = str(item[0]), str(item[1])
        pad_head = True if len(heading) < len(cell) else False
        pad = abs(len(heading) - len(cell))
        pad = whitespace[:pad]
        pad_left = pad[:len(pad)//2]
        pad_right = pad[len(pad)//2:]
        if pad_head:
            heading = pad_left + heading + pad_right
        else:
            cell = pad_left + cell + pad_right
        headings += [heading]
        cells += [cell]

    border, head, body = '', '', ''

    for i in range(len(item_tuples)):
        temp_head = f'| {headings[i]} '
        temp_body = f'| {cells[i]} '
        border += border_pattern[:len(temp_head)]
        head += temp_head
        body += temp_body
        if i == len(item_tuples) - 1:
            head += '|'
            body += '|'
            border += '+'

    print(border)
    print(head)
    print(border)
    print(body)
    print(border)
    print(' ')


def time_since(started):
    """
    Return a string representing the elapsed time since 'started'.
    Args:
        started (float): Start time (from time.time()).
    Returns:
        str: Elapsed time as a string.
    """
    elapsed = time.time() - started
    m = int(elapsed // 60)
    s = int(elapsed % 60)
    if m >= 60:
        h = int(m // 60)
        m = m % 60
        return f'{h}h {m}m {s}s'
    else:
        return f'{m}m {s}s'


def save_attention(attn, path):
    """
    Save an attention matrix as an image file.
    Args:
        attn (np.ndarray): Attention matrix.
        path (str): Output file path (without extension).
    """
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 6))
    plt.imshow(attn.T, interpolation='nearest', aspect='auto')
    fig.savefig(f'{path}.png', bbox_inches='tight')
    plt.close(fig)


def save_spectrogram(M, path, length=None):
    """
    Save a spectrogram as an image file.
    Args:
        M (np.ndarray): Spectrogram matrix.
        path (str): Output file path (without extension).
        length (int, optional): If provided, crop the spectrogram to this length.
    """
    import matplotlib.pyplot as plt
    M = np.flip(M, axis=0)
    if length:
        M = M[:, :length]
    fig = plt.figure(figsize=(12, 6))
    plt.imshow(M, interpolation='nearest', aspect='auto')
    fig.savefig(f'{path}.png', bbox_inches='tight')
    plt.close(fig)


def plot(array):
    """
    Plot a 1D array using matplotlib.
    Args:
        array (np.ndarray): Array to plot.
    """
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(30, 5))
    ax = fig.add_subplot(111)
    ax.xaxis.label.set_color('grey')
    ax.yaxis.label.set_color('grey')
    ax.xaxis.label.set_fontsize(23)
    ax.yaxis.label.set_fontsize(23)
    ax.tick_params(axis='x', colors='grey', labelsize=23)
    ax.tick_params(axis='y', colors='grey', labelsize=23)
    plt.plot(array)


def plot_spec(M):
    """
    Plot a spectrogram using matplotlib.
    Args:
        M (np.ndarray): Spectrogram matrix.
    """
    import matplotlib.pyplot as plt
    M = np.flip(M, axis=0)
    plt.figure(figsize=(18,4))
    plt.imshow(M, interpolation='nearest', aspect='auto')
    plt.show()

