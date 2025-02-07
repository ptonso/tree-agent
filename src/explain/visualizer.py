import numpy as np
import cv2
import matplotlib.cm as cm
from typing import List, Optional

class AutoencoderVisualizer:
    def __init__(self, 
                 target_width: int = 760, 
                 target_height: int = 480, 
                 embed_width: int = 40, 
                 row_spacing: int = 20,
                 obs_embed_spacing: int = 15,
                 embed_column_spacing: int = 3,
                 window_name: str = "Autoencoder Predictions"):
        """
        Initialize visualizer with specific dimensions.
        Args:
            target_width: Target width for the full visualization (default: 760px)
            target_height: Target height for the full visualization (default: 480px)
            embed_width: Width of the embedding visualization (20px per column)
            row_spacing: Pixels between rows (default: 20px)
            obs_embed_spacing: Pixels between observation and embedding (default: 15px)
            embed_column_spacing: Pixels between mu and logvar columns (default: 3px)
        """
        self.target_width = target_width
        self.target_height = target_height
        self.embed_width = embed_width // 2  # Split between mu and logvar
        self.row_spacing = row_spacing
        self.obs_embed_spacing = obs_embed_spacing
        self.embed_column_spacing = embed_column_spacing
        self.window_name = window_name
        
        total_spacing = (self.obs_embed_spacing * 2) + self.embed_column_spacing
        remaining_width = target_width - embed_width - total_spacing
        self.obs_width = remaining_width // 2

    def _resize_observation(self, img: np.ndarray) -> np.ndarray:
        """Resize observation/reconstruction to target dimensions."""
        h, w = img.shape[:2]
        aspect = w / h
        new_h = min(self.target_height // 2 - self.row_spacing, self.obs_width // aspect)
        new_w = int(new_h * aspect)
        
        if new_w > self.obs_width:
            new_w = self.obs_width
            new_h = int(new_w / aspect)
            
        return cv2.resize(img, (new_w, int(new_h)), interpolation=cv2.INTER_AREA)

    def _create_embed_visualization(self, embed: np.ndarray, target_height: int) -> np.ndarray:
        """
        Creates embedding visualization with mu and logvar as separate columns with different colormaps.
        """
        E = embed.shape[0] // 2
        mu, logvar = embed[:E], embed[E:]

        def normalize(x: np.ndarray) -> np.ndarray:
            min_val, max_val = x.min(), x.max()
            return (x - min_val) / (max_val - min_val + 1e-8) if max_val > min_val else np.zeros_like(x)

        mu_norm = normalize(mu)
        logvar_norm = normalize(logvar)

        # Calculate embed height (70% of observation height)
        embed_height = int(target_height * 0.7)
        padding_height = (target_height - embed_height) // 2

        mu_vis = cv2.resize(mu_norm.reshape(-1, 1), (self.embed_width, embed_height), 
                           interpolation=cv2.INTER_NEAREST)
        logvar_vis = cv2.resize(logvar_norm.reshape(-1, 1), (self.embed_width, embed_height), 
                               interpolation=cv2.INTER_NEAREST)

        mu_colored = (cm.viridis(mu_vis)[:, :, :3] * 255).astype(np.uint8)
        logvar_colored = (cm.plasma(logvar_vis)[:, :, :3] * 255).astype(np.uint8)

        top_padding = np.zeros((padding_height, self.embed_width, 3), dtype=np.uint8)
        bottom_padding = np.zeros((target_height - embed_height - padding_height, self.embed_width, 3), dtype=np.uint8)

        mu_column = np.vstack([top_padding, mu_colored, bottom_padding])
        logvar_column = np.vstack([top_padding, logvar_colored, bottom_padding])
        
        separator = np.zeros((target_height, self.embed_column_spacing, 3), dtype=np.uint8)
        
        embed_vis = np.hstack([mu_column, separator, logvar_column])
        
        return embed_vis

    def visualize(
            self, 
            observations: List[np.ndarray], 
            embeds: List[np.ndarray], 
            x_hats: List[np.ndarray], 
            obs_saliencies: Optional[List[np.ndarray]], 
        ) -> np.ndarray:
        """
        Visualizes a batch of observations, embeddings, and reconstructions.
        """
        rows = []
        for i, (obs, embed, x_hat) in enumerate(zip(observations, embeds, x_hats)):
            obs_resized = self._resize_observation(obs)
            x_hat_resized = self._resize_observation(x_hat)
            
            if obs_saliencies is not None and obs_saliencies[i] is not None:
                saliency_resized = cv2.resize(obs_saliencies[i], obs_resized.shape[1::-1], interpolation=cv2.INTER_NEAREST)
                
                saliency_resized = saliency_resized.mean(axis=-1) # Convert (H,W,3) -> (H,W)
                saliency_resized = (saliency_resized - saliency_resized.min()) / (saliency_resized.max() - saliency_resized.min() + 1e-8)
                saliency_colored = (cm.jet(saliency_resized)[:, :, :3] * 255).astype(np.uint8)
                obs_resized = cv2.addWeighted(obs_resized, 0.85, saliency_colored, 0.15, 0)

            embed_vis = self._create_embed_visualization(embed, obs_resized.shape[0])
            spacer = np.zeros((obs_resized.shape[0], self.obs_embed_spacing, 3), dtype=np.uint8)
            row_vis = np.hstack([obs_resized, spacer, embed_vis, spacer, x_hat_resized])
            
            padding_w = max(0, (self.target_width - row_vis.shape[1]) // 2)
            if padding_w > 0:
                padding = np.zeros((row_vis.shape[0], padding_w, 3), dtype=np.uint8)
                row_vis = np.hstack([padding, row_vis, padding])
            
            rows.append(row_vis)
            
            if len(observations) > 1:
                horizontal_spacer = np.zeros((self.row_spacing, row_vis.shape[1], 3), dtype=np.uint8)
                rows.append(horizontal_spacer)

        full_vis = np.vstack(rows)
        
        padding_h = max(0, (self.target_height - full_vis.shape[0]) // 2)
        if padding_h > 0:
            padding = np.zeros((padding_h, full_vis.shape[1], 3), dtype=np.uint8)
            full_vis = np.vstack([padding, full_vis, padding])

        return full_vis


    def render(
            self, 
            observations: List[np.ndarray], 
            embeds: List[np.ndarray], 
            x_hats: List[np.ndarray],
            obs_saliencies: Optional[List[np.ndarray]] = None
        ) -> None:
        full_vis = self.visualize(observations, embeds, x_hats, obs_saliencies)
        cv2.imshow(self.window_name, full_vis)
        cv2.waitKey(1)
            

if __name__ == "__main__":
    obs1 = np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)
    obs2 = np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)
    x_hat1 = np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)
    x_hat2 = np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)
    embed1 = np.random.randn(128)  # 64 mu + 64 log_sigma
    embed2 = np.random.randn(128)

    visualizer = AutoencoderVisualizer(
        obs_embed_spacing=15,
        embed_column_spacing=3,
        row_spacing=20
    )
    visualizer.visualize([obs1, obs2], [embed1, embed2], [x_hat1, x_hat2], wait_time=0)