import cv2
import math
import numpy as np
import random
from imgaug import augmenters as ia

def rotate_image(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result

def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )

def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]

def augmentate(blob, scale=False, crop=False, hflip=False, vflip=False, gamma=False,
               contrast=False, brightness=False, rotate=False, shear=False):
    """Perform data-augmentations on all modalities of an image blob.

    Args:
        if set, every argument is prepented with a individual probability that determines
            whether or not the given augmentation is performed
        scale (list of 2 values or False): Scale the image by a factor from the given
            interval
        crop (int or False): crop the image (after scaling) to a square of the given size
        vflip (bool): vertically flip the image
        hflip (bool): horizontally flip the image
        gamma (list of 2 values or False): apply gamma correction/noise with a random
            factor from the given interval
    Returns:
        augemted image blob
    """
    modalities = list(blob.keys())

    # find out whether or not we are doing cropping later on
    do_crop = False
    if crop and crop[0] > random.random():
        do_crop = True

    if scale and do_crop and scale[0] > random.random():
        h, w = blob[modalities[0]].shape[:2]
        min_scale = crop[1] / float(min(h, w))
        k = random.uniform(max(min_scale, scale[1]), scale[2])

        # RGB is resized using bilinear interpolation, all other modalities should use
        # nearest neighbour as their values do not necessarily behave like rgb
        if 'rgb' in blob:
            blob['rgb'] = cv2.resize(blob['rgb'], None, fx=k, fy=k)
        for m in (m for m in modalities if m != 'rgb'):
            blob[m] = cv2.resize(blob[m], None, fx=k, fy=k,
                                 interpolation=cv2.INTER_NEAREST)

    if rotate and rotate[0] > random.random():
        h, w = blob[modalities[0]].shape[:2]
        deg = np.random.randint(rotate[1], rotate[2])
        rect = largest_rotated_rect(w, h, math.radians(deg))
        for m in modalities:
            blob[m] = crop_around_center(rotate_image(blob[m], deg), *rect)

    if shear and do_crop and shear[0] > random.random():
        h, w = blob[modalities[0]].shape[:2]
        shear_px = np.random.randint(shear[1] * w, shear[2] * w) \
            * np.random.choice([-1, 1])
        print(shear_px)
        augmentation = ia.Affine(shear=shear_px)
        for m in modalities:
            blob[m] = augmentation.augment_image(blob[m])

    if do_crop:
        h, w = blob[modalities[0]].shape[:2]
        h_c = random.randint(0, h - crop[1])
        w_c = random.randint(0, w - crop[1])
        for m in modalities:
            blob[m] = blob[m][h_c:h_c+crop[1], w_c:w_c+crop[1], ...]

    if hflip and hflip > random.random() and np.random.choice([0, 1]):
        for m in modalities:
            blob[m] = np.flip(blob[m], axis=0)

    if vflip and vflip > random.random() and np.random.choice([0, 1]):
        for m in modalities:
            blob[m] = np.flip(blob[m], axis=1)

    if contrast and 'rgb' in modalities:
        augmentation = ia.Sometimes(contrast[0], ia.ContrastNormalization((contrast[1],
                                                                           contrast[2])))
        blob['rgb'] = augmentation.augment_image(blob['rgb'])

    if brightness and 'rgb' in modalities:
        augmentation = ia.Sometimes(brightness[0],
                                    ia.Add((brightness[1], brightness[2])))
        blob['rgb'] = augmentation.augment_image(blob['rgb'])

    if gamma and 'rgb' in modalities and gamma[0] > random.random():
        # gamma noise should only be applied to rgb
        k = random.uniform(gamma[1], gamma[2])
        lut = np.array([((i / 255.0) ** (1/k)) * 255
                        for i in np.arange(0, 256)]).astype("uint8")
        blob['rgb'] = lut[blob['rgb']]

    return blob

def crop_multiple(data, multiple_of=16):
    """Force the array dimension to be multiple of the given factor.

    Args:
        data: a >=2-dim array, first 2 dims will be cropped
        multiple_of: the factor, as an int
    Returns:
        cropped array
    """
    h, w = data.shape[0], data.shape[1]
    h_c, w_c = [d - (d % multiple_of) for d in [h, w]]
    if h_c != h or w_c != w:
        return data[:h_c, :w_c, ...]
    else:
        return data
