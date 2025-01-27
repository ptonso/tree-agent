import cv2
import numpy as np
import matplotlib.cm as cm
from typing import Optional, List
from src.agent.structures import Action
from src.run.config import Config

class IntegratedVisualizer:
    def __init__(self, config: Config, action_labels: Optional[List[str]] = None):
        """Initialize visualizer with combined game and action probability display.
        
        Args:
            action_labels: List of action names to display
        """
        self.config = config
        self.action_labels = action_labels or ['Forward', 'Turn Left', 'Turn Right']
        self.action_dim = len(self.action_labels)
        
        # Colors for action probability bars (in BGR)
        self.colors = [
            (128, 128, 78),  # Teal
            (108, 128, 128),  # Blue-green
            (108, 108, 158)   # Purple-blue
        ]

        self.highlight_color = (0, 165, 255)
        
    def create_action_overlay(self, 
                              action_probs: np.ndarray, 
                              sampled_action: int, 
                              ) -> np.ndarray:
        """Create visualization of action probabilities.
        
        Args:
            action_probs: (action_dim,)
            sampled_action: Index of the selected action
            width: Width of the overlay
            height: Height of the overlay
        Returns:
            np.ndarray: BGR image array of shape (height, width, 3)
        """
        width = self.config.env.render_width
        height = self.config.env.render_height // 4
        overlay = np.ones((height, width, 3), dtype=np.uint8) * 255 # white bg
        
        label_height = 20
        spacing = 10
        box_height = height - label_height - 2 * spacing
        box_width = (width - spacing * (self.action_dim + 1)) // self.action_dim
        
        for i in range(self.action_dim):
            prob = action_probs[i]

            r, g, b, _ = cm.viridis(prob)
            color = (int(b*255), int(g*255), int(r*255))

            x1 = spacing + i * (box_width + spacing)
            y1 = label_height + spacing
            x2 = x1 + box_width
            y2 = y1 + box_height

            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            
            # Highlight selected action
            if i == sampled_action:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), self.highlight_color, 2)
            
            label = f"{self.action_labels[i]}: {prob:.2f}"
            cv2.putText(overlay, label, (x1, label_height - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        return overlay

    def combine_displays(self, 
                         game_frame: object, 
                         action: Action
                         ) -> np.ndarray:
        """Combine game frame with action probability visualization.
        Args:
            game_frame: RGB/BGR frame from the game
            action: Action object
        """
        action_probs = action.as_numpy[0][:self.action_dim]
        sampled_action = action.sampled_action

        action_overlay = self.create_action_overlay(
            action_probs,
            sampled_action,
        )
        
        combined_frame = np.vstack([
            game_frame,
            action_overlay
        ])
        
        return combined_frame