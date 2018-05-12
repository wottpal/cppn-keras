# -*- coding: utf-8 -*-

import os
import time
import math 
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image
from skimage.color import hsv2rgb
import matplotlib.pyplot as plt
from keras import models, layers, initializers


def build_model(variance, bw = False, depth = 32):
    """Builds and returns CPPN."""
    input_shape=(4,)
    init = initializers.VarianceScaling(scale=variance)

    model = models.Sequential()
    model.add(layers.Dense(depth, kernel_initializer=init, activation='tanh', input_shape=input_shape))
    model.add(layers.Dense(depth, kernel_initializer=init, activation='tanh'))
    model.add(layers.Dense(depth, kernel_initializer=init, activation='tanh'))
    model.add(layers.Dense(1 if bw else 3, activation='tanh'))
    
    model.compile(optimizer='rmsprop', loss='mse')
    return model


def create_grid(x_dim, y_dim, scale = 1.0):
    N = np.mean((x_dim, y_dim))
    x = np.linspace(- x_dim / N * scale, x_dim / N * scale, x_dim)
    y = np.linspace(- y_dim / N * scale, y_dim / N * scale, y_dim)

    X, Y = np.meshgrid(x, y)

    x = np.ravel(X).reshape(-1, 1)
    y = np.ravel(Y).reshape(-1, 1)
    r = np.sqrt(x ** 2 + y ** 2)

    return x, y, r


def create_image(model, x, x_dim, y, y_dim, r):
    lat = np.random.normal(0,1,1)
    Z = np.repeat(lat, x.shape[0]).reshape(-1, x.shape[0])
    X = np.concatenate([x, y, r, Z.T], axis=1)

    pred = model.predict(X)

    img = []
    channels = pred.shape[1]
    for channel in range(channels):
        yp = pred[:, channel]
        yp = (yp - yp.min()) / (yp.max()-yp.min())
        img.append(yp.reshape(y_dim, x_dim))
    img = np.dstack(img)

    if channels == 3: img = hsv2rgb(img)
    img = (img * 255).astype(np.uint8)

    return img


def plot_images(images):
    """Plots the given images with pyplot (max 9)."""
    n = min(len(images), 9)
    rows = int(math.sqrt(n))
    cols = n // rows
    fig = plt.figure()
    for i in range(1, n+1):
        image = images[i-1]
        fig.add_subplot(rows, cols, i)
        plt.axis("off")
        plt.imshow(image)
    plt.show()


def save_image(image, results_dir, postfix = ""):
    """Saves given image-array under the given path."""
    if not os.path.exists(results_dir): os.makedirs(results_dir)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    image_name = f"img.{timestr}{postfix}.png"
    image_path = os.path.join(results_dir, image_name)
    file = Image.fromarray(image)
    file.save(image_path)    
    
    return image_path
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--save', action='store_const', const=True)
    parser.add_argument('-p', '--plot', action='store_const', const=True)
    parser.add_argument('--n', type=int, nargs='?', default=1)
    parser.add_argument('--path', type=str, nargs='?', default="./results")
    parser.add_argument('--x', type=int, nargs='?', default=500)
    parser.add_argument('--y', type=int, nargs='?', default=500)
    parser.add_argument('--bw', action='store_const', const=True)
    parser.add_argument('--variance', type=int, nargs='?')
    args = parser.parse_args()

    images = []
    for _ in tqdm(range(args.n)):
        x, y, r = create_grid(args.x, args.y, 1.0)

        variance = args.variance or np.random.uniform(50, 150)
        model = build_model(variance, bw = args.bw)

        image = create_image(model, x, args.x, y, args.y, r)
        image = image.squeeze()
        images.append(image)

        if args.save: 
            image_path = save_image(image, args.path, f'.var{variance:.0f}')
            tqdm.write(f"Image saved under {image_path}")

    if args.plot: plot_images(images)

