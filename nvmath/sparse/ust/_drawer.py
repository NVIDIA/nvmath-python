# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0

"""
This module defines drawing methods for the universal sparse tensor (UST).
"""

__all__ = []

import math

import numpy as np

from ._converters import TensorDecomposer


class RequiredPackageError(Exception):
    pass


WHITE = (0xFF, 0xFF, 0xFF)
BLACK = (0x00, 0x00, 0x00)
GRAY = (0xF0, 0xF0, 0xF0)
RED = (0xFF, 0x80, 0x80)
GREEN = (0x80, 0xFF, 0x80)
BLUE = (0x80, 0x80, 0xFF)
YELLOW = (0xFF, 0xFF, 0x80)
PURPLE = (0xFF, 0x08, 0xFF)


def _pos_length(tensor, i):
    if tensor.pos(i) is not None:
        return tensor.pos(i).size
    return 0


def _crd_length(tensor, i):
    if tensor.crd(i) is not None:
        return tensor.crd(i).size
    return 0


def _max_length(tensor):
    length = tensor.nse
    for i in range(tensor.num_levels):
        length = max(length, _pos_length(tensor, i))
        length = max(length, _crd_length(tensor, i))
    return length


def _scale_font(txt, w):
    try:
        from PIL import ImageFont  # type: ignore[import-not-found]
    except ImportError as e:
        raise RequiredPackageError("The PIL package is required for visualizing UST.") from e

    length = len(txt)
    f = max(8, w - length - 1)
    off = 2 + max(0, 4 - length) * w // 4
    try:
        font = ImageFont.truetype("UbuntuMono-B.ttf", f)
    except:
        font = ImageFont.load_default(size=f)
    return off, font


def draw_tensor(tensor, name=None):
    """
    Draws tensor contents (1D, 2D, 3D).

    This method is useful to illustrate tensor contents of smaller examples.

    Args:
        tensor: tensor to draw
        name: filename to save to if not None
    Returns:
        Image, can be displayed with show()
    """

    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore[import-not-found]
    except ImportError as e:
        raise RequiredPackageError("The PIL package is required for visualizing UST.") from e

    if tensor.num_dimensions == 1:
        M = 1
        N = tensor.extents[0]  # as row vector
        K = 1
        L1 = None
        L2 = "i"
        L3 = None
    elif tensor.num_dimensions == 2:
        M = tensor.extents[0]
        N = tensor.extents[1]
        K = 1
        L1 = "i"
        L2 = "j"
        L3 = None
    elif tensor.num_dimensions == 3:
        M = tensor.extents[1]
        N = tensor.extents[2]
        K = tensor.extents[0]  # i is batch, drawn horizontally
        L1 = "j"
        L2 = "k"
        L3 = "i"
    else:
        raise TypeError(f"Cannot draw {tensor.num_dimensions}-dimensional tensor")

    B = 8  # border
    W = 32  # width unit
    S = W // 4  # shadow unit
    F = W // 2  # font unit
    X = W + 2 * B + (S + S + N * W) * K - S
    Y = W + 2 * B + S + M * W

    try:
        font = ImageFont.truetype("UbuntuMono-B.ttf", F)
    except:
        font = ImageFont.load_default(size=F)

    img = Image.new("RGB", (X, Y), color=WHITE)
    draw = ImageDraw.Draw(img)

    def drawGrid(k):
        xo = k * (N * W + S + S)
        xs = B + W + xo
        ys = B + W
        xe = xs + N * W
        ye = ys + M * W
        draw.rectangle([xs + S, ys + S, xe + S, ye + S], fill=BLACK, outline=BLACK, width=1)
        draw.rectangle([xs, ys, xe, ye], fill=WHITE, outline=BLACK, width=1)
        x = xs
        for _ in range(1, N):
            x = x + W
            draw.line([x, ys, x, ye], fill=BLACK, width=1)
        y = ys
        for _ in range(1, M):
            y = y + W
            draw.line([xs, y, xe, y], fill=BLACK, width=1)

    def drawGridCell(i, j, k, txt, color=GRAY):
        xo = k * (N * W + S + S)
        xs = B + W + j * W + xo
        ys = B + W + i * W
        xe = xs + W
        ye = ys + W
        if False:  # conditional cell coloring goes here
            color = RED
        draw.rectangle([xs, ys, xe, ye], fill=color, outline=BLACK, width=1)
        off, vfont = _scale_font(txt, F)
        draw.text([xs + off, ys + S], txt, fill=BLACK, font=vfont)

    def visit123d(idx, val):
        if len(idx) == 1:
            drawGridCell(0, idx[0], 0, str(val))
        elif len(idx) == 2:
            drawGridCell(idx[0], idx[1], 0, str(val))
        else:
            drawGridCell(idx[1], idx[2], idx[0], str(val))

    # Draw coordinate system.
    draw.line([B, B, B + W, B + W], fill=BLACK, width=3)
    if L1 is not None:
        draw.text([B, B + F], L1, fill=BLACK, font=font)
        for i in range(M):
            draw.text([B + S, F + (i + 1) * W], str(i), fill=BLACK, font=font)
    if L2 is not None:
        draw.text([B + F + S, B], L2, fill=BLACK, font=font)
        for i in range(N):
            draw.text([F + (i + 1) * W, B + S], str(i), fill=BLACK, font=font)
    if L3 is not None:
        for i in range(K):
            draw.text([B + W + i * (N * W + S + S), 0], f"{L3}={i}", fill=BLACK, font=font)

    for k in range(K):
        drawGrid(k)

    TensorDecomposer(tensor, visit123d).run()

    if name is not None:
        img.save(name)
    return img


def draw_tensor_storage(tensor, name=None):
    """
    Draws tensor storage (UST).

    This method is useful to illustrate the UST storage of smaller examples.

    Args:
        tensor: tensor to draw
        name: filename to save to if not None
    Returns:
        Image, can be displayed with show()
    """

    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore[import-not-found]
    except ImportError as e:
        raise RequiredPackageError("The PIL package is required for visualizing UST.") from e

    M = tensor.num_levels * 2 + 1
    N = _max_length(tensor)
    B = 8  # border
    W = 32  # width unit
    S = W // 4  # shadow unit
    F = W // 2  # font unit
    X = 2 * B + N * W + S + 2 * W
    Y = 2 * B + M * (W + S + S) - S

    try:
        font = ImageFont.truetype("UbuntuMono-B.ttf", F)
    except:
        font = ImageFont.load_default(size=F)

    img = Image.new("RGB", (X, Y), color=WHITE)
    draw = ImageDraw.Draw(img)

    def drawRow(i, txt, length, color=WHITE):
        xs = B
        ys = B + i * (W + S + S)
        draw.text([xs, ys + S], txt, fill=BLACK, font=font)
        xs = xs + 2 * W
        xe = xs + (S if length == 0 else length * W)
        ye = ys + W
        draw.rectangle([xs + S, ys + S, xe + S, ye + S], fill=BLACK, outline=BLACK, width=1)
        draw.rectangle([xs, ys, xe, ye], fill=color, outline=BLACK, width=1)
        for _ in range(1, length):
            xs = xs + W
            draw.line([xs, ys, xs, ye], fill=BLACK, width=1)

    def drawRowCell(i, j, txt, color=BLACK):
        xs = B + 2 * W + j * W
        ys = B + i * (W + S + S)
        off, vfont = _scale_font(txt, F)
        if False:  # conditional cell coloring goes here
            draw.rectangle([xs, ys, xs + W, ys + W], fill=RED, outline=BLACK, width=1)
        draw.text([xs + off, ys + S], txt, fill=color, font=vfont)

    for level in range(tensor.num_levels):
        p = _pos_length(tensor, level)
        drawRow(2 * level, f"pos[{level}]", p)
        for j in range(p):
            drawRowCell(2 * level, j, str(tensor.pos(level).tensor[j].item()))
        c = _crd_length(tensor, level)
        drawRow(2 * level + 1, f"crd[{level}]", c)
        for j in range(c):
            drawRowCell(2 * level + 1, j, str(tensor.crd(level).tensor[j].item()))
    drawRow(M - 1, "values", tensor.nse)
    for j in range(tensor.nse):
        drawRowCell(M - 1, j, str(tensor.val.tensor[j].item()))

    if name is not None:
        img.save(name)
    return img


def draw_tensor_raw(tensor, name=None):
    """
    Draws tensor nonzero structure (2D, 3D).

    This method scales to larger tensors.

    Args:
        tensor: tensor to draw
        name: filename to save to if not None
    Returns:
        Image, can be displayed with show()
    """
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore[import-not-found]
    except ImportError as e:
        raise RequiredPackageError("The PIL package is required for visualizing UST.") from e

    if tensor.num_dimensions == 2:
        m, n, k = tensor.extents[1], tensor.extents[0], 0  # X goes down
    elif tensor.num_dimensions == 3:
        m, n, k = tensor.extents[0], tensor.extents[2], tensor.extents[1]  # Z goes up
    else:
        raise TypeError(f"Cannot draw raw {tensor.num_dimensions}-dimensional tensor")

    S = 0
    while (m >> S) > 512:
        S += 1
    while (n >> S) > 512:
        S += 1
    while (k >> S) > 512:
        S += 1
    X = min(m, m >> S)
    Y = min(n, n >> S)
    Z = min(k, k >> S)
    if ((m >> S) << S) != m:
        X += 1
    if ((n >> S) << S) != n:
        Y += 1
    if ((k >> S) << S) != k:
        Z += 1
    F = 16

    try:
        font = ImageFont.truetype("UbuntuMono-B.ttf", F)
    except:
        font = ImageFont.load_default(size=F)

    img = Image.new("RGB", (X, Y), color=WHITE)
    draw = ImageDraw.Draw(img)

    if tensor.num_dimensions == 2:

        def visit2d(idx, val):
            img.putpixel((idx[1] >> S, idx[0] >> S), BLACK)

        draw.line([0, 0, X - 1, 0], fill=RED, width=1)
        draw.line([0, 0, 0, Y - 1], fill=RED, width=1)
        draw.line([X - 1, 0, X - 1, Y - 1], fill=RED, width=1)
        draw.line([0, Y - 1, X - 1, Y - 1], fill=RED, width=1)

        TensorDecomposer(tensor, visit2d).run()

        draw.text([0, Y / 2], "I", fill=BLUE, font=font, stroke_width=1.2)
        draw.text([X / 2, 0], "J", fill=BLUE, font=font, stroke_width=1.2)
    else:
        # Fixed rotation around center in X/Z plane.
        XR = 0
        ZR = Z

        # Fixed eye for projection.
        D = Z * 2

        # Rotation and translation (variable).
        PHI = 45
        XO = X // 2
        YO = Y // 2
        ZO = Z // 2

        def project(x, y, z):
            # Translate into view.
            x = x - XO
            y = y - YO
            z = z + ZO
            # Rotate [x,z,1].
            co = math.cos(PHI)
            si = math.sin(PHI)
            xr = co * x - si * z + ((1 - co) * XR + si * ZR)
            zr = si * x + co * z + ((1 - co) * ZR - si * XR)
            x = xr
            y = y
            z = zr
            # Project (if not in plane).
            if z == -D:
                x = x
                y = y
            else:
                x = x * (D / (z + D))
                y = y * (D / (z + D))
            # Translate back.
            x = int(x + XO)
            y = int(y + YO)
            return x, y, 0

        def line(x1, y1, z1, x2, y2, z2):
            (xp1, yp1, zp) = project(x1, y1, z1)
            (xp2, yp2, zp) = project(x2, y2, z2)
            draw.line([xp1, Y - yp1, xp2, Y - yp2], fill=RED, width=3)

        def visit3d(idx, val):
            x, y, z = project(idx[0] >> S, idx[2] >> S, idx[1] >> S)
            # Guarded put pixel.
            if x >= 0 and x < X and y >= 0 and y < Y:
                img.putpixel((x, Y - y - 1), BLACK)

        line(0, 0, 0, X, 0, 0)
        line(0, 0, 0, 0, 0, Z)
        line(X, 0, 0, X, 0, Z)
        line(0, 0, Z, X, 0, Z)
        line(0, Y, 0, X, Y, 0)
        line(0, Y, 0, 0, Y, Z)
        line(X, Y, 0, X, Y, Z)
        line(0, Y, Z, X, Y, Z)
        line(0, 0, 0, 0, Y, 0)
        line(X, 0, 0, X, Y, 0)
        line(0, 0, Z, 0, Y, Z)
        line(X, 0, Z, X, Y, Z)

        TensorDecomposer(tensor, visit3d).run()

        (x, y, z) = project(X / 2, 0, 0)
        draw.text([x, Y - y - 1], "I", fill=BLUE, font=font, stroke_width=1.2)
        (x, y, z) = project(0, 0, Z / 2)
        draw.text([x, Y - y - 1], "J", fill=BLUE, font=font, stroke_width=1.2)
        (x, y, z) = project(0, Y / 2, 0)
        draw.text([x, Y - y - 1], "K", fill=BLUE, font=font, stroke_width=1.2)

    if name is not None:
        img.save(name)
    return img


def animate_tensor(tensor, name=None):
    """
    Animates tensor nonzero structure (3D).

    This method scales to larger tensors.

    Args:
        tensor: tensor to animate
        name: filename to save to if not None
    Returns:
        HTML if name is None (to embed in other output)
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
        from IPython.display import HTML  # type: ignore
        from matplotlib.animation import FuncAnimation, PillowWriter  # type: ignore
    except ImportError as e:
        raise RequiredPackageError("Animation packages are required for visualizing UST.") from e

    if tensor.num_dimensions == 3:
        X, Y, Z = tensor.extents[0], tensor.extents[1], tensor.extents[2]
    else:
        raise TypeError(f"Cannot animate {tensor.num_dimensions}-dimensional tensor")

    # Map largest to [0,1].
    U = float(max(X, Y, Z))
    X = X / U
    Y = Y / U
    Z = Z / U

    points = np.empty((tensor.nse, 3), dtype=np.float32)
    index = 0

    def visit3d(idx, val):
        nonlocal index
        x, y, z = idx[0], idx[1], idx[2]
        points[index, 0] = x / U
        points[index, 1] = y / U
        points[index, 2] = z / U
        index += 1

    TensorDecomposer(tensor, visit3d).run()

    # Set up frame.
    cube_vertices = np.array([[0, 0, 0], [X, 0, 0], [X, Y, 0], [0, Y, 0], [0, 0, Z], [X, 0, Z], [X, Y, Z], [0, Y, Z]])
    cube_edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]])
    fig = plt.figure(figsize=(5, 5), dpi=72)
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.cla()
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)

        # Set the animation view.
        elev = 45
        azim = frame * 2  # degrees per frame
        ax.view_init(elev=elev, azim=azim)

        # Draw cube edges in red.
        for edge in cube_edges:
            v = cube_vertices[edge]
            ax.plot3D(v[:, 0], v[:, 1], v[:, 2], color="red", linewidth=3, alpha=1)

        # Draw points.
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c="black", s=0.15, alpha=0.4, depthshade=True)

        # Add labels at each axis
        ax.text(X / 2, 0, 0, "I", fontsize=12, fontweight="bold", color="blue")
        ax.text(0, Y / 2, 0, "J", fontsize=12, fontweight="bold", color="blue")
        ax.text(0, 0, Z / 2, "K", fontsize=12, fontweight="bold", color="blue")

        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    anim = FuncAnimation(fig, update, frames=180, interval=50)
    plt.close()

    if name is not None:
        anim.save(name, writer=PillowWriter(fps=20))
    else:
        return HTML(anim.to_html5_video())


def draw_network(tensors, name=None):
    """
    Draws network consisting of linear layer weight matrices.

    Args:
        tensor: list of tensors to draw
        name: filename to save to if not None
    Returns:
        Image, can be displayed with show()
    """

    try:
        from PIL import Image, ImageDraw  # type: ignore[import-not-found]
    except ImportError as e:
        raise RequiredPackageError("The PIL package is required for visualizing UST.") from e

    CELLS = [tensor.shape[0] for tensor in tensors]
    CELLS.append(tensors[-1].shape[1])
    B = 8  # border
    W = 32  # width unit
    C1 = 4  # cell spacing start
    C2 = W - C1  # cell spacing end
    CO = W // 2
    CL = len(CELLS)
    CM = max(CELLS)
    D = CM * W
    X = 2 * B + D * (CL - 1) + W
    Y = 2 * B + D

    img = Image.new("RGB", (X, Y), color=WHITE)
    draw = ImageDraw.Draw(img)

    def drawCircle(i, cl, color=BLUE):
        m = (CM - CELLS[cl]) / 2
        x = B + cl * D
        y = B + (i + m) * W
        draw.ellipse((x + C1, y + C1, x + C2, y + C2), width=2, fill=color, outline=BLACK)

    def drawLine(i, j, cl, color=BLACK):
        m1 = (CM - CELLS[cl]) / 2
        m2 = (CM - CELLS[cl + 1]) / 2
        xs = B + cl * D + CO
        ys = B + (i + m1) * W + CO
        xe = xs + D
        ye = B + (j + m2) * W + CO
        draw.line([xs, ys, xe, ye], fill=color, width=1)

    for cl in range(0, CL):
        for i in range(0, CELLS[cl]):
            drawCircle(i, cl)

    for cl in range(0, CL - 1):

        def visit(tensor, transposed, lvl):
            def visit2d(idx, val):
                if transposed:
                    drawLine(idx[1], idx[0], lvl)
                else:
                    drawLine(idx[0], idx[1], lvl)

            TensorDecomposer(tensor, visit2d).run()

        visit(tensors[cl].ust, tensors[cl].transposed, cl)

    if name is not None:
        img.save(name)
    return img
