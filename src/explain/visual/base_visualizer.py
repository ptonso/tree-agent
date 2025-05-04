
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class VisEmbed:
    image: np.ndarray # final color image of shape (H,W,3)
    width: int
    height: int


class BaseVisualizer:
    """
    Abstract class for all visualizers.
    Provides standard interface for rendering and updating visualizations.
    """

    def __init__(
            self,
            config: "Config",
        ):
        self.window_name = config.window_name
        self.width = config.window_width
        self.height = config.window_height
        self.config = config

        self.canvas = np.full(
            (self.height, self.width, 3),
            self.config.bgc, dtype=np.uint8
            )

    def resize_image(
            self, 
            width: Optional[int] = None, 
            height: Optional[int] = None, 
            image: Optional[np.ndarray] = None
            ) -> None:
        """Resizes image.
        If no image, resize and update canvas
        """
        if width is not None:
            self.width = width
        if height is not None:
            self.height = height
        if image is not None:
            image = np.array(image, dtype=np.uint8)
            return cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_AREA)

        self.canvas = cv2.resize(self.canvas, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return self.canvas

    def get_numpy(self) -> np.ndarray:
        """Return current canvas a NumPy array."""
        return self.canvas
    
    def render(self) -> None:
        """
        Displays the current visualization in an OpenCV window.
        """
        cv2.imshow(self.window_name, self.canvas)
        cv2.waitKey(1)

    def save(self, output_dir: str = "reports") -> None:
        """
        Saves the current visualization to a file.
        The filename is generated based on the current timestamp.
        """
        import os
        import time
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{output_dir}/{self.window_name}_{int(time.time())}.png"
        cv2.imwrite(filename, self.canvas)
        print(f"Saved visualization to {filename}")

    def blend(
            self,
            weight: float,
            values1: np.ndarray,
            values2: Optional[np.ndarray] = None
        )-> np.ndarray:
        """Blend value1 with value2 based on weight.
        If no value2, blend background color."""
       
        if values2 is None:
            # background
            values2 = np.array(self.config.bgc, dtype=np.float32) # e.g. (255,255,255)

        weight = np.array(weight).astype(np.float32)
        values1 = values1.astype(np.float32)
        values2 = values2.astype(np.float32)

        if weight.ndim == values1.ndim - 1:
            weight = weight[..., None]

        # color_i = (w_i * value1) + ( (1-w_i) * values2)
        blended = weight * values1 + (1 - weight) * values2
        return blended.astype(np.uint8)


    def build_background(self, width: int, height: int) -> np.ndarray:
        return np.full((height, width, 3), self.config.bgc, dtype=np.uint8)

    def build_embed(
        self,
        embed: np.ndarray,
        box_size: int = 80
    ) -> VisEmbed:
        """
        Builds a color image representation of a 1D embed array.
        - If embed is a perfect square (N*N), arrange it as N×N.
        - Otherwise, put it in a single row (length×1).
        """
        flat = embed.flatten()
        length = flat.size

        side = int(np.sqrt(length))
        if side * side == length:
            rows, cols = side, side
        else:
            rows, cols = 1, length

        max_dim = max(rows, cols)
        cell_size = max(1, box_size // max_dim)

        out_h = rows * cell_size
        out_w = cols * cell_size
        out_img = self.build_background(out_w, out_h)

        max_val = np.max(np.abs(flat)) + 1e-8

        for i in range(length):
            val = flat[i]
            alpha = min(1.0, max(0.0, abs(val) / max_val))

            blue = np.array(self.config.blue, dtype=np.float32)
            color = self.blend(alpha, blue)

            r, c = divmod(i, cols)
            sy = r * cell_size
            sx = c * cell_size
            cv2.rectangle(
                out_img,
                (sx, sy),
                (sx + cell_size, sy + cell_size),
                tuple(int(x) for x in color),
                -1
            )

        return VisEmbed(image=out_img, width=out_w, height=out_h)



    def update(self, *args, **kwargs) -> None:
        """
        Updates the visualization dynamically.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError
