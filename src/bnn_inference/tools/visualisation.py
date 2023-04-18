import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import squarify
import torchvision
from PIL import Image, ImageDraw, ImageFont
from sklearn.decomposition import PCA


def get_treemap_grid(array_num_samples, list_label=None, aspect_ratio=1.33, sort=True):
    """
    :param array_num_samples:
    :param aspect_ratio:
    :param sort: If True, the array_num_samples is sorted and large rectangles are allocated from left upper.
    :return:
    """
    if list_label is None:
        list_label = range(len(array_num_samples))

    array_num_samples = np.array(array_num_samples)

    sum_samples = 1.0 * np.sum(array_num_samples)  # 1.1 is margin
    h = int(math.sqrt(sum_samples / aspect_ratio))
    w = int(math.sqrt(aspect_ratio * sum_samples))
    sum_samples = h * w

    treemap_grid = -5 * np.ones((h, w), dtype=np.int)

    if sort:
        list_label = list_label[np.flip(np.argsort(array_num_samples))]
        array_num_samples = np.flip(np.sort(array_num_samples))

    array_num_samples_norm = squarify.normalize_sizes(
        array_num_samples, w, h
    )  # Normalize is necessary
    rects = squarify.squarify(array_num_samples_norm, 0, 0, w, h)

    for i_classes in range(len(array_num_samples)):
        tmp_label = list_label[i_classes]

        tmp_w_min = math.floor(rects[i_classes]["x"])
        tmp_h_min = math.floor(rects[i_classes]["y"])
        tmp_w_max = math.ceil((rects[i_classes]["x"] + rects[i_classes]["dx"]))
        tmp_h_max = math.ceil((rects[i_classes]["y"] + rects[i_classes]["dy"]))

        # prevent disappearing of small class
        if tmp_w_min == tmp_w_max:
            tmp_w_min -= 1
        if tmp_h_min == tmp_h_max:
            tmp_h_min -= 1

        treemap_grid[tmp_h_min:tmp_h_max, tmp_w_min:tmp_w_max] = tmp_label

    idxs_edge = -np.ones(
        (len(array_num_samples), 2, 2), dtype=np.int
    )  # [h_min, w_min], [h_max, w_max]
    # find left up index and right down index
    # the order of list idxs_edge is the saa as list_label
    for i_classes in range(len(array_num_samples)):
        tmp_idxs = np.argwhere(treemap_grid == list_label[i_classes])
        if len(tmp_idxs) == 0:
            print()
        idxs_edge[list_label[i_classes], 0, :] = tmp_idxs[0]
        idxs_edge[list_label[i_classes], 1, :] = tmp_idxs[-1]

    return treemap_grid, idxs_edge


def patch_allocation_pca(w, h, samples):
    """
    To visualize patches in 2d tiles, allocate samples by PCA order
    :param w: output grid width
    :param h: output grid height
    :param samples: num_samples * num_features. num_samples should be approximately (w * h).
    :return: 2d numpy array. the values are row index of samples.
    """
    # assert math.fabs(samples.shape[0] - w * h) < 0.2 * samples.shape[0], \
    #     'number of samples is largely different from the nubmer of allocated slots.' \
    #     + str(samples.shape[0]) + str(w * h)

    num_samples = samples.shape[0]
    if num_samples >= 2:
        pca = PCA(n_components=2)
        embs = pca.fit_transform(samples)
    else:
        embs = np.zeros((1, 2), dtype=np.int)
    # numsamples

    if h > w:
        principal = h
        secondary = w
    else:
        principal = w
        secondary = h

    if secondary == 0:
        print()

    # longer axis should be the principal direction
    ret_idx_matrix = -np.ones((secondary, principal), dtype=np.int)
    idx_sort_principal = np.argsort(embs[:, 0])
    # for the case num_samples > w*h

    for i_principal in range(min(math.ceil(num_samples / secondary), principal)):
        if i_principal == int(num_samples / secondary):
            tmp_num_samples = num_samples - i_principal * secondary
        else:
            tmp_num_samples = secondary
        tmp_idx_principal = idx_sort_principal[
            i_principal * secondary : i_principal * secondary + tmp_num_samples
        ]
        tmp_samples_in_bin = embs[tmp_idx_principal, :]
        idx_sort_secondary = np.argsort(tmp_samples_in_bin[:, 1])
        ret_idx_matrix[0:tmp_num_samples, i_principal] = tmp_idx_principal[
            idx_sort_secondary
        ]

    if h > w:
        ret_idx_matrix = ret_idx_matrix.transpose()

    return ret_idx_matrix


def add_frame_to_image(input_img, colour, frame_width_ratio=0.03):
    """

    :type input_img: object
    :param input_img: values in each pixels should be float values between 0 and 1
    :param colour:
    :param frame_width:
    :return:
    """
    frame_width = math.ceil(frame_width_ratio * input_img.shape[0])
    # input_img = np.copy(np.asarray(input_img))
    colour_frame_w = np.tile(colour, (frame_width, input_img.shape[1], 1))
    colour_frame_h = np.tile(colour, (input_img.shape[0], frame_width, 1))
    # input_img.flags.writeable = True
    input_img[0:frame_width, :, :] = colour_frame_w
    input_img[(-frame_width):, :, :] = colour_frame_w
    input_img[:, 0:frame_width, :] = colour_frame_h
    input_img[:, (-frame_width):, :] = colour_frame_h
    return input_img


def get_nine_samples_pil_image(
    config,
    base_path,
    data_frame,
    samples_latents,
    patch_size=227,
    colour=None,
    num_samples_per_side=3,
    remove_edge=True,
    return_image=True,
):
    """

    :param data_frame: panda DataFrame instance which contains the full path of images in the column named 'image file name'.
    :param samples_latents: number_of_samples * number_of_features numpy array. The feature vectors.
    :param patch_size: int. the width and height of the patch cropped from original images.
    :param colour: 3 * 1 arrays. [r g b] and each values should be float value between 0 and 1.
    :param num_samples_per_side: The output will be tiled image that have num_samples_per_side patches on each edge.
    :param remove_edge: If True, the samples on the edge of distribution will not be selected so the result looks more stable.
    :return: PIL Image Object.
    """

    header = config.csv_reader.headers

    num_sample = samples_latents.shape[0]
    w = int(math.sqrt(num_sample))
    h = w
    tf_crop = torchvision.transforms.CenterCrop(patch_size)
    alloc_mat = patch_allocation_pca(w, h, samples_latents)
    # avoid edge
    if remove_edge and w - num_samples_per_side >= 2:
        w = w - 2
        h = h - 2
        alloc_mat = alloc_mat[1:-1, 1:-1]

    if return_image:
        # load PIL
        ret_np = np.zeros(
            (patch_size * num_samples_per_side, patch_size * num_samples_per_side, 3),
            dtype=np.float,
        )
    else:
        ret_array_idx = -np.ones(
            (num_samples_per_side, num_samples_per_side), dtype=np.uint
        )

    for i_h in range(num_samples_per_side):
        for i_w in range(num_samples_per_side):
            tmp_h = int(i_h * ((h - 1) / (num_samples_per_side - 1.0)))
            tmp_w = int(i_w * ((w - 1) / (num_samples_per_side - 1.0)))
            tmp_idx = alloc_mat[tmp_h, tmp_w]

            print(tmp_idx, base_path)

            print(data_frame[header.relative_path])

            tmp_filepath = (
                Path(base_path) / data_frame[header.relative_path].iloc[tmp_idx]
            )

            if return_image:
                tmp_img = tf_crop(Image.open(tmp_filepath))
                tmp_img = np.array(tmp_img) / 255.0
                if colour is not None:
                    tmp_img = add_frame_to_image(tmp_img, colour=colour)
                ret_np[
                    i_h * patch_size : (i_h + 1) * patch_size,
                    i_w * patch_size : (i_w + 1) * patch_size,
                    :,
                ] = tmp_img
            else:
                ret_array_idx[i_h, i_w] = tmp_idx

    if return_image:
        return Image.fromarray(np.uint8(ret_np * 255))
    else:
        return ret_array_idx


# create tile

# return PIL object


def get_clustering_tile_pil_image(
    config,
    base_path,
    data_frame,
    samples_latents,
    patch_size_org=227,
    max_num_patches=10000,
    resize_rate=0.1,
    list_colour=None,
    draw_label=True,
    result_label="clustering result",
):
    """

    :param draw_label: True for drawing the class index on the image
    :param data_frame: pandas.Dataframe which contains 'clustering result' and 'image file name' in its key. 'clustering result' should be int value start from 0. 'image file name' should be file fullpath of original image.
    :param samples_latents: (num_samples * num_features) 2D numpy array
    :param patch_size_org: default 224
    :param max_num_patches: default 10000
    :param resize_rate: default 0.1.
    :return: PIL object of the result image
    """
    header = config.csv_reader.headers
    # samples dummy
    if samples_latents is None:
        samples_latents = np.random.randn(len(data_frame), 2)

    # downsampling patches. proportion of each group in original dataset is preserved.
    if len(data_frame) > max_num_patches:
        labels = pd.Series.value_counts(data_frame[result_label], sort=False).index
        value_ratio_org = np.array(
            pd.Series.value_counts(data_frame[result_label], sort=False, normalize=True)
        )

        list_effective_idxs = []
        for i_label in range(len(labels)):
            idx_tmp_label_all = data_frame[
                data_frame[result_label] == labels[i_label]
            ].index
            idx_tmp_label_effective = random.sample(
                list(idx_tmp_label_all),
                k=int(max_num_patches * value_ratio_org[i_label]),
            )
            list_effective_idxs.extend(idx_tmp_label_effective)

        data_frame = data_frame.iloc[list_effective_idxs].reset_index(drop=True)
        samples_latents = samples_latents[list_effective_idxs, :]
    labels = pd.Series.value_counts(data_frame[result_label], sort=False).index
    value_counts = np.array(
        pd.Series.value_counts(data_frame[result_label], sort=False)
    )

    # get treemap grid
    treemap_grid, edge_idxs = get_treemap_grid(value_counts, list_label=labels)
    patch_size_resized = int(patch_size_org * resize_rate)
    result = (
        np.ones(
            (
                patch_size_resized * treemap_grid.shape[0],
                patch_size_resized * treemap_grid.shape[1],
                3,
            ),
            dtype="float16",
        )
        * 0.3
    )  # default colour is grey
    tf_crop_resize = torchvision.transforms.Compose(
        [
            torchvision.transforms.CenterCrop(patch_size_org),
            torchvision.transforms.Resize(patch_size_resized),
        ]
    )
    if list_colour is None:
        list_colour = sns.color_palette("hls", len(labels))
    for i_label in range(len(labels)):
        tmp_label = labels[i_label]
        tmp_lu_idx = edge_idxs[tmp_label, 0, :]
        tmp_rd_idx = edge_idxs[tmp_label, 1, :]
        tmp_w = tmp_rd_idx[1] - tmp_lu_idx[1] + 1
        tmp_h = tmp_rd_idx[0] - tmp_lu_idx[0] + 1

        tmp_idx = (data_frame.index[data_frame[result_label] == tmp_label]).to_list()
        tmp_img_filepaths = data_frame.iloc[tmp_idx][header.relative_path].reset_index(
            drop=True
        )

        tmp_samples = samples_latents[tmp_idx, :]
        ret = patch_allocation_pca(tmp_w, tmp_h, tmp_samples)

        # fill with patch
        for i_width in range(tmp_w):
            for i_height in range(tmp_h):
                sample_idx = ret[i_height, i_width]
                if sample_idx < 0:
                    tmp_image = 0.3 * np.ones(
                        (patch_size_resized, patch_size_resized, 3), dtype=np.float
                    )  # default colour is grey
                else:
                    tmp_image = tf_crop_resize(
                        Image.open(Path(base_path) / tmp_img_filepaths[sample_idx])
                    )
                    tmp_image = np.array(tmp_image) / 255.0

                tmp_image = add_frame_to_image(tmp_image, list_colour[tmp_label])
                result[
                    (tmp_lu_idx[0] + i_height)
                    * patch_size_resized : (tmp_lu_idx[0] + i_height + 1)
                    * patch_size_resized,
                    (tmp_lu_idx[1] + i_width)
                    * patch_size_resized : (tmp_lu_idx[1] + i_width + 1)
                    * patch_size_resized,
                    :,
                ] = tmp_image

        # #     draw bold line for separation
        # separate_line_width = 10
        # result[tmp_lu_idx[0] * patch_size:tmp_rd_idx[0] * patch_size-separate_line_width,tmp_lu_idx[1]*patch_size::separate_line_width,: ]=
    # compress_w = int(resize_rate * result.shape[1])
    # compress_h = int(resize_rate * result.shape[0])
    result_pil = Image.fromarray(np.uint8(result * 255))
    # processing to pillow type
    draw = ImageDraw.Draw(result_pil, mode="RGB")
    for i_label in range(len(labels)):
        tmp_label = labels[i_label]
        tmp_lu_idx = edge_idxs[tmp_label, 0, :]
        tmp_rd_idx = edge_idxs[tmp_label, 1, :]
        vertex_lu_y = tmp_lu_idx[0] * patch_size_resized
        vertex_lu_x = tmp_lu_idx[1] * patch_size_resized
        vertex_rd_y = (tmp_rd_idx[0] + 1) * patch_size_resized - 1
        vertex_rd_x = (tmp_rd_idx[1] + 1) * patch_size_resized - 1
        frame_width = int(patch_size_resized * 0.1)
        tmp_colour_255 = tuple(
            (np.array(list_colour[tmp_label]) * 255).astype(np.uint8).tolist()
        )
        draw.rectangle(
            ((vertex_lu_x, vertex_lu_y), (vertex_rd_x, vertex_rd_y)),
            fill=None,
            width=frame_width,
            outline=tmp_colour_255,
        )
        # draw label
        if draw_label:
            font_size = int(result.shape[0] * 0.05)
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf", font_size
            )
            # text_label = 'class ' + str(tmp_label)
            text_label = str(tmp_label)
            draw.text(
                (vertex_lu_x + frame_width, vertex_lu_y + frame_width),
                text_label,
                fill=tmp_colour_255,
                font=font,
            )

        # resize before save
    # result_pil = result_pil.resize((compress_w, compress_h))
    return result_pil


if __name__ == "__main__":
    # for test treemap_grid
    # array_num_samples = [4252, 6381, 2977, 421, 60, 1256, 8954, 4902]
    # treemap_grid = get_treemap_grid(array_num_samples)
    # treemap_grid
    #
    # plt.imshow(treemap_grid)
    # plt.show()
    # for i in range(-1, len(array_num_samples), 1):
    #     print(np.sum(treemap_grid == i))

    #         for test pca sort
    w = 100
    h = 120
    samples = np.random.rand(13000, 10)
    idx_matrix = patch_allocation_pca(w, h, samples)
    idx_matrix
