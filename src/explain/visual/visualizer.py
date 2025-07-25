import cv2
import numpy as np
import torch
from typing import Optional

from src.explain.visual.base_visualizer import BaseVisualizer
from src.explain.visual.tree_visualizer import SoftTreeVisualizer
from src.explain.visual.vae_visualizer import VAEVisualizer

from src.run.config import VisConfig

class Visualizer(BaseVisualizer):
    """
    Combines a tree visualizer and a VAE visualizer into one unified canvas.
    """
    def __init__(
            self,
            window_name: str = "Visualizer"
            ):
        
        vis_config = VisConfig()
        self.tree_visualizer = SoftTreeVisualizer(vis_config.tree)
        self.vae_visualizer = VAEVisualizer(vis_config.vae)
        self.mode = vis_config.vae.mode
        self.config = vis_config.overall
        
        super().__init__(self.config)

    def update(
            self,
            observation: np.ndarray,
            state: "State",
            decoded: np.ndarray,
            world_model: Optional["WorldModel"] = None,
            saliency: Optional[np.ndarray] = None,
            tree: Optional["SoftDecisionTree"] = None
        ) -> None:

        full_background = self._create_background()

        if self.mode == "full":
            mid = full_background.shape[1] // 2
            vae_bg = full_background[:, :mid, :].copy()
            tree_bg = full_background[:, mid:, :].copy()
        elif self.mode == "actual":
            mid = full_background.shape[0] // 2
            vae_bg = full_background[:mid, :, :].copy()
            tree_bg = full_background[mid:, :, :].copy()
        else:
            raise ValueError(f"Unkown mode: {self.mode}")

        self.vae_visualizer.update(
            observation, state, decoded,
            world_model=world_model, 
            saliency=saliency,
            # background=vae_bg
            )
        self.tree_visualizer.update(
            embed=state.state_data, 
            tree=tree, 
            world_model=world_model,
            # background=tree_bg
            )
        
        tree_canvas = self.tree_visualizer.get_numpy()
        vae_canvas = self.vae_visualizer.get_numpy()

        if self.mode == "full":
            # merge side by side
            target_height = max(tree_canvas.shape[0], vae_canvas.shape[0])
            tree_canvas_resized = self.tree_visualizer.resize_image(width=tree_canvas.shape[1], height=target_height)
            vae_canvas_resized  = self.vae_visualizer.resize_image(width=vae_canvas.shape[1], height=target_height)
            
            self.canvas = np.hstack([vae_canvas_resized, tree_canvas_resized])

        elif self.mode == "actual":

            # merge top down
            target_width = max(tree_canvas.shape[1], vae_canvas.shape[1])
            tree_canvas_resized = self.tree_visualizer.resize_image(width=target_width, height=tree_canvas.shape[0])
            vae_canvas_resized  = self.vae_visualizer.resize_image(width=target_width, height=vae_canvas.shape[0])

            self.canvas = np.vstack([vae_canvas_resized, tree_canvas_resized])

        self.canvas = self.resize_image(
            width=self.config.window_width, 
            height=self.config.window_height
            )
        
    
    def _create_background(self) -> np.ndarray:
        """Create a shared gradient background."""
        h, w = self.config.window_height, self.config.window_width
        gradient = np.linspace(100, 230, h, dtype=np.uint8)[:, np.newaxis]
        gradient = np.repeat(gradient, w, axis=1)

        blue = np.clip(gradient + 25, 0, 255)
        green = np.clip(gradient - 30, 0, 255)
        red = np.clip(gradient - 50, 0, 255)

        return cv2.merge([blue, green, red])




if __name__ == "__main__":
    import torch
    import numpy as np
    from src.run.config import SoftConfig
    from src.explain.soft_tree import SoftDecisionTree
    from src.agent.structures import State

    torch.manual_seed(42)
    config = SoftConfig()

    input_dim = 64
    output_dim = 3
    sample_x = torch.randn(1, input_dim)
    tree = SoftDecisionTree(config)
    tree.setup_network(sample_x, torch.randn(1, output_dim))
        
    visualizer = Visualizer(window_name="Combined Visualization")
    
    dummy_tree_input = sample_x
    dummy_observation = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
    dummy_decoded = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
    dummy_embedding = np.random.randn(64)
    dummy_state = State(dummy_embedding, device="cuda")

    visualizer.update(
        dummy_observation,
        dummy_state,
        dummy_decoded,
        world_model=None,
        tree=tree
    )

    visualizer.render()
