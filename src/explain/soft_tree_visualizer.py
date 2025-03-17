from dataclasses import dataclass
import cv2
import numpy as np
import torch
import matplotlib.cm as cm
from typing import Optional, Tuple, List, Dict
import math

# Assuming these are in your codebase
from src.run.config import SoftConfig
from src.explain.soft_tree import SoftDecisionTree, InnerNode, LeafNode

@dataclass
class VisualizerConfig:
    window_width: int = 1400  # Enough width for better spacing
    window_height: int = 1000 # Enough height for vertical spacing
    node_panel_base_size: int = 80
    embed_square_base_size: int = 10
    embed_square_size: int = 10
    embed_square_spacing: int = 1
    border_thickness_scale: float = 4.0
    node_vertical_spacing: int = 180  
    split_line_thickness: int = 2
    font_scale: float = 0.6  
    font_thickness: int = 1
    histogram_height: int = 60  
    horizontal_bar_spacing: int = 4  
    embedding_arrangement: str = "2D"  # "1D" or "2D"
    window_name: str = "Soft Decision Tree"

class SoftDecisionTreeVisualizer:
    def __init__(self, config: VisualizerConfig, tree: SoftDecisionTree):
        self.config = config
        self.tree = tree
        self.tree_depth = tree.max_depth
        self.node_positions = {}

        # Dynamically size the node panels based on tree depth
        self.node_panel_size = self.config.node_panel_base_size + 10 * self.tree_depth
        self.embed_square_size = max(5, self.config.embed_square_base_size - self.tree_depth)

    def render(self, X: torch.Tensor) -> None:
        """
        Generates and displays the visualization for a given input X.
        """
        if self.tree.root is None:
            raise ValueError("Tree is not initialized. Call `setup_network` first.")

        # Convert input to NumPy for coloring squares by embedding value
        input_embedding = X.cpu().numpy().flatten()

        # Create blank canvas
        canvas = np.ones(
            (self.config.window_height, self.config.window_width, 3), 
            dtype=np.uint8
        ) * 255

        # Recursively layout and draw the tree from the root
        self._layout_and_draw_tree(
            canvas,
            node=self.tree.root,
            embed=input_embedding,
            x=self.config.window_width // 2,
            y=50,           # Start a bit down from top
            depth=1
        )

        # Display the result
        cv2.imshow(self.config.window_name, canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _layout_and_draw_tree(
        self, 
        canvas: np.ndarray, 
        node, 
        embed: np.ndarray, 
        x: int, 
        y: int, 
        depth: int
    ) -> None:
        """
        Recursively calculates positions for children & draws edges/nodes.
        """
        if isinstance(node, LeafNode):
            self._draw_leaf(canvas, node, x, y)
            return

        if isinstance(node, InnerNode):
            # Probability of going right child
            device = next(node.fc.parameters()).device
            embed_tensor = torch.tensor(embed, dtype=torch.float32, device=device).unsqueeze(0)
            prob_right = node.forward(embed_tensor).item()

            # Adjust horizontal spacing
            level_width = int(self.config.window_width / (2 ** (depth - 0.5)))
            child_y = y + self.config.node_vertical_spacing

            left_x = x - (level_width // 2)
            right_x = x + (level_width // 2)

            # Draw edges first
            self._draw_edge(canvas, x, y + 15, left_x, child_y - 15, 1 - prob_right)
            self._draw_edge(canvas, x, y + 15, right_x, child_y - 15, prob_right)

            # Draw the node panel
            self._draw_inner_node(canvas, node, embed, x, y)

            # Recurse to children
            self._layout_and_draw_tree(canvas, node.left, embed, left_x, child_y, depth + 1)
            self._layout_and_draw_tree(canvas, node.right, embed, right_x, child_y, depth + 1)

    def _draw_edge(
        self, 
        canvas: np.ndarray, 
        x1: int, 
        y1: int, 
        x2: int, 
        y2: int, 
        prob: float
    ) -> None:
        """
        Draws an edge between two nodes with probability annotation.
        """
        color = (255, 0, 0) if prob > 0.5 else (0, 0, 255)
        thickness = int(self.config.split_line_thickness + prob * 3)

        cv2.line(canvas, (x1, y1), (x2, y2), color, thickness)

        # Probability text near midpoint
        mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2 - 10
        cv2.putText(
            canvas, 
            f"{prob:.2f}", 
            (mid_x, mid_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.font_scale,
            (0, 0, 0),
            self.config.font_thickness
        )

    def _draw_inner_node(
        self, 
        canvas: np.ndarray, 
        node: InnerNode, 
        embed: np.ndarray, 
        x: int, 
        y: int
    ) -> None:
        """
        Draw an inner node that shows:
          - The input embedding as color for squares
          - The node's weight as the border thickness
        """
        # Node has fc.weight -> shape [1, 64], node.beta -> shape [1]
        # We multiply them to get a final 64-d weight vector
        fc_weight = node.fc.weight.detach().cpu().numpy().flatten()  # shape = (64,)
        beta_scalar = node.beta.detach().cpu().item()                # single float

        final_weight = fc_weight * beta_scalar  # shape=(64,)

        # Build a 64-square panel, 
        # color from 'embed', border from 'final_weight'
        squares_info = self._build_embedding_grid(embed, final_weight)

        panel_w, panel_h = squares_info["panel_w"], squares_info["panel_h"]
        tl_x = x - panel_w // 2
        tl_y = y - panel_h // 2  

        # Draw panel border
        cv2.rectangle(
            canvas, 
            (tl_x, tl_y), 
            (tl_x + panel_w, tl_y + panel_h),
            (0, 0, 0),
            2
        )

        # Now draw squares
        for sq in squares_info["squares"]:
            sx = tl_x + sq["rel_x"]
            sy = tl_y + sq["rel_y"]
            fill_color = sq["fill_color"]
            border_thick = sq["border_thick"]

            cv2.rectangle(canvas, (sx, sy), (sx + sq["size"], sy + sq["size"]), fill_color, -1)
            cv2.rectangle(canvas, (sx, sy), (sx + sq["size"], sy + sq["size"]), (0, 0, 0), border_thick)

    def _draw_leaf(
        self, 
        canvas: np.ndarray, 
        node: LeafNode, 
        x: int, 
        y: int
    ) -> None:
        """
        Draw leaf node as a horizontal histogram of action probabilities.
        """
        panel_w = self.node_panel_size
        panel_h = int(self.node_panel_size * 0.7)

        tl_x, tl_y = x - panel_w // 2, y

        cv2.rectangle(
            canvas, 
            (tl_x, tl_y), 
            (tl_x + panel_w, tl_y + panel_h),
            (0, 0, 0),
            2
        )

        action_probs = torch.softmax(node.param, dim=0).detach().cpu().numpy()
        n_actions = len(action_probs)
        bar_width = (panel_w - (n_actions - 1) * self.config.horizontal_bar_spacing) // n_actions

        offset_x = tl_x
        offset_y = tl_y + panel_h - 2

        for i, prob in enumerate(action_probs):
            bar_len = int(prob * self.config.histogram_height)
            cv2.rectangle(
                canvas,
                (offset_x, offset_y - bar_len),
                (offset_x + bar_width, offset_y),
                (0, 0, 255),
                -1
            )
            offset_x += bar_width + self.config.horizontal_bar_spacing

    def _build_embedding_grid(
        self, 
        embed: np.ndarray, 
        weight: np.ndarray
    ) -> Dict[str, object]:
        """
        Creates a grid layout (1D or 2D) of squares, each with:
         - `fill_color` from the embedding value
         - `border_thick` from the node's weight
        """
        emb_len = len(embed)

        # Check if weight matches embed length
        if len(weight) != emb_len:
            print(f"⚠️ Warning: Weight length ({len(weight)}) != Embedding length ({emb_len}). Fixing shape...")
            if len(weight) == 1:
                weight = np.full(emb_len, weight[0])  
            else:
                weight = np.resize(weight, emb_len)

        # 1D or 2D arrangement
        if self.config.embedding_arrangement == "2D":
            side = int(math.sqrt(emb_len))
            rows, cols = (side, side) if side * side == emb_len else (1, emb_len)
        else:
            rows, cols = (1, emb_len)

        # Normalize embedding -> color
        normalized_embed = (embed - np.min(embed)) / (np.max(embed) - np.min(embed) + 1e-8)
        # Normalize weight -> border
        abs_weight = np.abs(weight)
        max_w = np.max(abs_weight) + 1e-8
        normalized_weight = abs_weight / max_w

        # Convert embed to color squares
        colors = (cm.viridis(normalized_embed)[:, :3] * 255).astype(np.uint8)
        colors = [tuple(map(int, c)) for c in colors]

        # Convert weight to thickness
        thicknesses = [1 + int(self.config.border_thickness_scale * nw) for nw in normalized_weight]

        # Compute grid size
        sq_size = self.config.embed_square_size
        sq_spacing = self.config.embed_square_spacing
        grid_w = cols * (sq_size + sq_spacing) - sq_spacing
        grid_h = rows * (sq_size + sq_spacing) - sq_spacing
        panel_w = max(self.config.node_panel_base_size, grid_w + 10)
        panel_h = max(self.config.node_panel_base_size, grid_h + 10)

        # Build squares
        squares = []
        margin_x = (panel_w - grid_w) // 2
        margin_y = (panel_h - grid_h) // 2

        for i in range(emb_len):
            r, c = divmod(i, cols)
            sx = margin_x + c * (sq_size + sq_spacing)
            sy = margin_y + r * (sq_size + sq_spacing)

            squares.append({
                "rel_x": sx,
                "rel_y": sy,
                "size": sq_size,
                "fill_color": colors[i],
                "border_thick": thicknesses[i]
            })

        return {
            "panel_w": panel_w,
            "panel_h": panel_h,
            "squares": squares
        }

if __name__ == "__main__":
    # Example usage
    torch.manual_seed(42)
    input_dim = 64
    output_dim = 3
    X_sample = torch.randn(1, input_dim)

    # Build a SoftDecisionTree
    config = SoftConfig()
    ddt = SoftDecisionTree(config)
    ddt.setup_network(X_sample, torch.randn(1, output_dim))

    # Visualization config
    vis_config = VisualizerConfig(
        window_width=1400,
        window_height=1200,
        node_vertical_spacing=200,
        histogram_height=70,
        embedding_arrangement="2D"  # Show as a grid for 64 dims
    )

    # Render
    visualizer = SoftDecisionTreeVisualizer(vis_config, ddt)
    visualizer.render(X_sample)
