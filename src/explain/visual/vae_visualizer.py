from dataclasses import dataclass
import cv2
import numpy as np
import torch
import matplotlib.cm as cm
from typing import List, Tuple, Optional, Literal

from src.agent.structures import State
from src.explain.visual.base_visualizer import BaseVisualizer
from src.run.config import VAEVisualizerConfig



class VAEVisualizer(BaseVisualizer):
    """
    Visualizes outputs from a variational autoencoder.
    'full' experiment with latent variations
    'actual' show only the actual state.
    """
    def __init__(self, config: VAEVisualizerConfig):
        super().__init__(config)
        self.config = config
        self.variation_step = 0
        self.variation_direction = 1
    
    def update(
        self,
        observation: np.ndarray,
        state: State,
        decoded: np.ndarray,
        world_model: Optional["WorldModel"] = None,
        saliency: Optional[np.ndarray] = None
        ) -> None:
        embedding = state.for_render
        
        if self.config.saliency_mode and saliency is not None:
            with torch.no_grad():
                observation = self._apply_saliency_overlay(observation, saliency)

        if self.config.mode == "full":
            if not (world_model and state):
                raise ValueError("Full mode requires world_model")
            self._update_full(observation, embedding, decoded, world_model, state)
        elif self.config.mode == "actual":
            self._update_actual(observation, embedding, decoded)
        else:
            raise ValueError("Mode must be 'full' or 'actual'")
        

    def _update_full(
        self,
        observation: np.ndarray,
        embedding: np.ndarray,
        decoded: np.ndarray,
        world_model: "WorldModel",
        state: State
    ) -> None:
        variations = self._generate_latent_variations(world_model, state)
        varied_embeddings, varied_decoded, changed_dims = variations
        
        main_row = self._build_main_row(observation, embedding, decoded)
        variation_grid = self._build_variation_grid(varied_embeddings, varied_decoded, changed_dims)
        
        spacer = np.full((10, max(main_row.shape[1], variation_grid.shape[1]), 3), self.config.bgc, dtype=np.uint8)
        final_width = max(main_row.shape[1], variation_grid.shape[1])
        
        main_row = self.resize_image(image=main_row, width=final_width)
        variation_grid = self.resize_image(image=variation_grid, width=final_width)
        
        visualization = np.vstack([main_row, spacer, variation_grid])

        self.canvas = self._add_margin(visualization)
        

    def _update_actual(
        self,
        observation: np.ndarray,
        embedding: np.ndarray,
        decoded: np.ndarray
    ) -> None:
        main_row = self._build_main_row(observation, embedding, decoded)
        self.canvas = self._add_margin(main_row)
        

    def _apply_saliency_overlay(
            self,
            observation: np.ndarray,
            saliency_map: np.ndarray
        ) -> np.ndarray:
        """Convert an image to grayscale and overlays the saliency map as a heatmap"""
        grayscale = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        grayscale = cv2.cvtColor(grayscale, cv2.COLOR_GRAY2RGB)

        saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
        saliency_colormap = (cm.jet(saliency_map)[:, :, :3] * 255).astype(np.uint8)

        overlay = cv2.addWeighted(grayscale, 0.5, saliency_colormap, 0.5, 0)
        return overlay

    def _add_margin(self, image: np.ndarray) -> np.ndarray:
        """Adds a top and lateral margin to the image."""
        top = self.config.top_margin
        lateral = self.config.lateral_margin
        return cv2.copyMakeBorder(
            image, 
            top=top, bottom=top, left=lateral, right=lateral,
            borderType=cv2.BORDER_CONSTANT, value=self.config.bgc
        )

    def _generate_latent_variations(
        self,
        world_model: "WorldModel",
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
        
        total_width = self.config.window_width
        spacer_width = 10
        emb_width = self.config.embedding_width
        image_width = (total_width - emb_width - 2 * spacer_width) // 2
        height = self.config.main_height

        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)        
        obs_resized = self.resize_image(image=observation, height=height, width=image_width)
        dec_resized = self.resize_image(image=decoded, height=height, width=image_width)
        
        emb_vis = self._visualize_embedding(embedding, height=emb_width)
        emb_vis = self.resize_image(image=emb_vis, width=emb_width, height=emb_width)  # Fix mismatch

        centered_square = self.build_background(width=emb_width, height=height)
        square_y_offset = (height - emb_width) // 2
        centered_square[square_y_offset:square_y_offset + emb_width, :, :] = emb_vis
        emb_vis = centered_square

        spacer = self.build_background(width=spacer_width, height=height)
        return np.hstack([obs_resized, spacer, emb_vis, spacer, dec_resized])


    def _build_variation_grid(
        self,
        embeddings: List[np.ndarray],
        decoded_images: List[np.ndarray],
        changed_dims: List[int]
    ) -> np.ndarray:
        var_grid_height = self.config.window_height - self.config.main_height
        cell_height = var_grid_height // 3
        cell_width = self.config.window_width // 3
        row_spacer_height = 10
        embed_width = self.config.embedding_width

        grid_cells = []
        for i in range(9):
            if i < len(embeddings):                
                emb_vis = self._visualize_embedding(embeddings[i], height=cell_height)
                emb_vis = self.resize_image(image=emb_vis, width=embed_width, height=embed_width)

                centered_square = self.build_background(width=embed_width, height=cell_height)
                square_y_offset = (cell_height - embed_width) // 2
                centered_square[square_y_offset:square_y_offset + embed_width, :, :] = emb_vis
                emb_vis = centered_square

                dec_resized = self.resize_image(
                    image=decoded_images[i], 
                    width=cell_width - embed_width - 10,
                    height=cell_height
                )
                col_spacer = self.build_background(10, cell_height)
                cell = np.hstack([emb_vis, col_spacer, dec_resized])
            else:
                cell = self.build_background(cell_width, cell_height)
            grid_cells.append(cell)

        row1 = np.hstack(grid_cells[0:3])
        row2 = np.hstack(grid_cells[3:6])
        row3 = np.hstack(grid_cells[6:9])

        # Ensure all rows have the same width
        final_width = max(row1.shape[1], row2.shape[1], row3.shape[1])

        row1 = self.resize_image(image=row1, width=final_width)
        row2 = self.resize_image(image=row2, width=final_width)
        row3 = self.resize_image(image=row3, width=final_width)
        row_spacer = self.build_background(final_width, row_spacer_height)

        grid_with_spacers = np.vstack([row1, row_spacer, row2, row_spacer, row3])

        return grid_with_spacers


    def _visualize_embedding(
        self,
        embedding: np.ndarray,
        height: int,
        highlighted_dim: Optional[int] = None
    ) -> np.ndarray:
        vis_embed = self.build_embed(embedding, box_size=self.config.embedding_width)
        return vis_embed.image
        # flat_embedding = embedding.flatten()
        # min_val, max_val = flat_embedding.min(), flat_embedding.max()
        # normalized = (flat_embedding - min_val) / ((max_val - min_val) + 1e-8)

        # alpha = np.full_like(flat_embedding, 0.2, dtype=np.float32)
        # if highlighted_dim is not None and 0 <= highlighted_dim < len(alpha):
        #     alpha[highlighted_dim] = 1.0

        # colors = (cm.viridis(normalized)[:, :3] * 255).astype(np.float32)

        # color_map = self.blend(alpha, colors).reshape(-1, 1, 3)
        # return self.resize_image(
        #     image=color_map, 
        #     width=self.config.embedding_width, 
        #     height=height
        #     )




if __name__ == "__main__":
    config = VAEVisualizerConfig(
        window_width=760,
        main_height=200,
        var_grid_height=300,
        embedding_width=40,
        window_name="Autoencoder Dummy",
        mode="actual",
        saliency_mode=False
    )
    
    visualizer = VAEVisualizer(config)
    
    # Create dummy observation and decoded images as random RGB images.
    observation = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
    decoded = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
    
    # Create a dummy embedding vector (e.g., 64-dimensional)
    embedding = np.random.randn(64)
    
    visualizer.update(observation, embedding, decoded)
    visualizer.render()