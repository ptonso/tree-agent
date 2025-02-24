from dataclasses import dataclass
import cv2
import numpy as np
import torch
import matplotlib.cm as cm
from typing import List, Tuple, Optional, Literal
from src.agent.structures import State
from src.agent.world_model import WorldModel

@dataclass
class VisualizerConfig:
    window_width: int = 760
    main_height: int = 200
    var_grid_height: int = 300
    embedding_width: int = 40
    window_name: str = "Autoencoder"
    mode: Literal["full", "actual"] = "full"

class AutoencoderVisualizer:
    def __init__(self, config: VisualizerConfig):
        self.config = config
        self.variation_step = 0
        self.variation_direction = 1

    def render(
        self,
        observation: np.ndarray,
        embedding: np.ndarray,
        decoded: np.ndarray,
        world_model: Optional[WorldModel] = None,
        state: Optional[State] = None
    ) -> None:
        if self.config.mode == "full":
            if not (world_model and state):
                raise ValueError("Full mode requires world_model and state")
            self._render_full(observation, embedding, decoded, world_model, state)
        elif self.config.mode == "actual":
            self._render_actual(observation, embedding, decoded)
        else:
            raise ValueError("Mode must be 'full' or 'actual'")

    def _render_full(
        self,
        observation: np.ndarray,
        embedding: np.ndarray,
        decoded: np.ndarray,
        world_model: WorldModel,
        state: State
    ) -> None:
        variations = self._generate_latent_variations(world_model, state)
        varied_embeddings, varied_decoded, changed_dims = variations
        
        main_row = self._build_main_row(observation, embedding, decoded)
        variation_grid = self._build_variation_grid(varied_embeddings, varied_decoded, changed_dims)
        
        spacer = np.zeros((10, max(main_row.shape[1], variation_grid.shape[1]), 3), dtype=np.uint8)
        final_width = max(main_row.shape[1], variation_grid.shape[1])
        
        main_row = self._resize_image(main_row, width=final_width)
        variation_grid = self._resize_image(variation_grid, width=final_width)
        
        visualization = np.vstack([main_row, spacer, variation_grid])
        self._display_image(visualization)

    def _render_actual(
        self,
        observation: np.ndarray,
        embedding: np.ndarray,
        decoded: np.ndarray
    ) -> None:
        main_row = self._build_main_row(observation, embedding, decoded)
        self._display_image(main_row)

    def _generate_latent_variations(
        self,
        world_model: WorldModel,
        state: State,
        num_variations: int = 9,
        variation_magnitude: float = 0.5
    ) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
        latent_tensor = state.as_tensor
        dim_count = latent_tensor.shape[1]
        start_idx = (self.variation_step * num_variations) % dim_count
        end_idx = min(start_idx + num_variations, dim_count)

        varied_embeddings, decoded_images, modified_dims = [], [], []

        for dim_idx in range(start_idx, end_idx):
            modified = latent_tensor.clone().detach()
            modified[:, dim_idx] += self.variation_direction * variation_magnitude
            
            modified_state = State(modified.cpu().numpy(), device=state.device)
            varied_embeddings.append(modified_state.state_data)
            modified_dims.append(dim_idx)

            with torch.no_grad():
                decoded_images.append(world_model.decode(modified_state).for_render)

        self._update_variation_state(end_idx >= dim_count)
        return varied_embeddings, decoded_images, modified_dims

    def _update_variation_state(self, cycle_complete: bool) -> None:
        self.variation_step = 0 if cycle_complete else self.variation_step + 1
        if cycle_complete:
            self.variation_direction *= -1

    def _build_main_row(
        self,
        observation: np.ndarray,
        embedding: np.ndarray,
        decoded: np.ndarray
    ) -> np.ndarray:
        height = self.config.main_height
        obs_resized = self._resize_image(observation, height=height)
        dec_resized = self._resize_image(decoded, height=height)
        emb_vis = self._visualize_embedding(embedding, height=height)
        
        spacer = np.zeros((height, 5, 3), dtype=np.uint8)
        return np.hstack([obs_resized, spacer, emb_vis, spacer, dec_resized])

    def _build_variation_grid(
        self,
        embeddings: List[np.ndarray],
        decoded_images: List[np.ndarray],
        changed_dims: List[int]
    ) -> np.ndarray:
        cell_height = self.config.var_grid_height // 3
        cell_width = self.config.window_width // 3
        row_spacer_height = 10
        
        grid_cells = []
        for i in range(9):
            if i < len(embeddings):
                emb_vis = self._visualize_embedding(embeddings[i], cell_height, changed_dims[i])
                dec_resized = self._resize_image(
                    decoded_images[i],
                    width=cell_width - self.config.embedding_width - 10,
                    height=cell_height
                )
                col_spacer = np.zeros((cell_height, 10, 3), dtype=np.uint8)
                cell = np.hstack([emb_vis, col_spacer, dec_resized])
            else:
                cell = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
            grid_cells.append(cell)

        
        row1 = np.hstack(grid_cells[0:3])
        row2 = np.hstack(grid_cells[3:6])
        row3 = np.hstack(grid_cells[6:9])

        row_spacer = np.zeros((row_spacer_height, row1.shape[1], 3), dtype=np.uint8)
        grid_with_spacers = np.vstack([row1, row_spacer, row2, row_spacer, row3])

        return grid_with_spacers

    def _visualize_embedding(
        self,
        embedding: np.ndarray,
        height: int,
        highlighted_dim: Optional[int] = None
    ) -> np.ndarray:
        flat_embedding = embedding.flatten()
        min_val, max_val = flat_embedding.min(), flat_embedding.max()
        normalized = (flat_embedding - min_val) / ((max_val - min_val) + 1e-8)

        alpha = np.ones_like(flat_embedding, dtype=np.float32)
        if highlighted_dim is not None:
            alpha[:] = 0.2
            alpha[highlighted_dim] = 1.0

        colors = cm.viridis(normalized)[:, :3] * 255
        color_map = ((colors.T * alpha).T).astype(np.uint8).reshape(-1, 1, 3)

        return self._resize_image(color_map, width=self.config.embedding_width, height=height)

    def _resize_image(
        self,
        image: np.ndarray,
        width: Optional[int] = None,
        height: Optional[int] = None
    ) -> np.ndarray:
        current_height, current_width = image.shape[:2]
        new_width = width or current_width
        new_height = height or current_height
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    def _display_image(self, image: np.ndarray) -> None:
        cv2.imshow(self.config.window_name, image)
        cv2.waitKey(1)