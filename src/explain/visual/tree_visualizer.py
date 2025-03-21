import cv2
import numpy as np
import torch
import math
from dataclasses import dataclass
from typing import Any, List, Tuple, Optional, Dict

from src.run.logger import create_logger
from src.run.config import SoftConfig
from src.explain.soft_tree import SoftDecisionTree, InnerNode, LeafNode
from src.explain.visual.base_visualizer import BaseVisualizer
from src.agent.world_model import WorldModel
from src.agent.structures import State


@dataclass
class NodeLayout:
    width: int
    height: int
    x_positions: List[List[int]]
    level_widths: List[int]


class SoftTreeVisualizer(BaseVisualizer):
    """
    Visualizes a differentiable decision tree.
    """
    def __init__(self, config: "TreeVisualizerConfig") -> None:
        super().__init__(config)
        self.logger = create_logger("api.log")

        self.config = config
        self.canvas = np.full(
            (config.window_height, config.window_width, 3),
            config.bgc,
            dtype=np.uint8
        )
        self.tree: Optional[SoftDecisionTree] = None
        self.world_model: Optional[WorldModel] = None
        self.decoded_cache: Dict[InnerNode, np.ndarray] = {}
        self.embed = np.zeros((1, 64))


    def update(
            self, 
            embed: np.ndarray,
            tree: Optional[SoftDecisionTree] = None,
            world_model: Optional[WorldModel] = None
        ) -> None:
        """
        Updates the tree visualization dynamically.
        """
        self.canvas[:] = self.config.bgc
        
        self.world_model = world_model
        self.embed = embed
            
        if tree is not None:
            self._initialize_layout(tree)

        if not self.tree or not self.tree.root:
            return
            
        nodes_list = self._collect_nodes_bfs(self.tree.root)

        if self.world_model:
            self._batch_decode_and_resize(nodes_list)
        
        self._draw_all_edges(nodes_list)

        for node, depth, idx in nodes_list:
            self._draw_inner_node(node, depth, idx)


    def _initialize_layout(self, tree: SoftDecisionTree):
        """Everything that is needed to rebuild the layout
        given a new tree."""
        if tree.root is None:
            return
        
        self.tree = tree
        self.max_depth = self.tree.max_depth
        self.level_counts = self._compute_level_counts()
        self.layout = self._compute_layout()

        root_x, root_y = self._get_node_position(0, 0)
        offset_y = 150

        if self.config.show_embed:
            self._draw_input(
                self.embed,
                x_left=int(root_x * 0.2),
                y_top=max(30, root_y - offset_y),
            )

        if self.config.show_legend:
            self._draw_legend(
                x_left=int(root_x * 1.5),
                y_top=max(30, root_y - offset_y)
            )

    def _compute_level_counts(self) -> List[int]:
        counts = [0] * (self.max_depth + 2)
        def recurse(node: Any, depth: int) -> None:
            if node is None:
                return
            counts[depth] += 1
            if isinstance(node, InnerNode):
                recurse(node.left, depth + 1)
                recurse(node.right, depth + 1)

        recurse(self.tree.root, 0)
        return counts

    def _compute_layout(self) -> NodeLayout:
        num_leaves = self.level_counts[self.max_depth]
        
        usable_width = self.config.window_width - 2 * self.config.lateral_margin
        usable_height = self.config.window_height - 2 * self.config.top_margin

        max_nodes_at_any_level= max(self.level_counts)
        max_leaf_nodes = self.level_counts[self.max_depth]

        node_width = usable_width // max(max_nodes_at_any_level, 1)
        level_height = usable_height // max(self.max_depth + 1, 1)
        node_height = min(level_height * 0.6, node_width)

        level_widths = [int(node_width * (1 + (self.max_depth - level) * 0.15))
                        for level in range(self.max_depth + 1)]
        
        x_positions = []
        for level, count in enumerate(self.level_counts):
            if count == 0:
                x_positions.append([])
                continue
            if count == 1:
                x_positions.append([self.config.window_width // 2])
            else:
                spacing = usable_width / (count + 1)
                level_positions = [
                    int(self.config.lateral_margin + (i + 1) * spacing)
                    for i in range(count)
                ]
                x_positions.append(level_positions)


        return NodeLayout(
            width=int(node_width * 0.8),
            height=int(node_height),
            x_positions=x_positions,
            level_widths=level_widths
        )
    
    def _collect_nodes_bfs(self, root: Any) -> List[Tuple[Any, int, int]]:
        queue = [(root, 0, 0)]
        out = []
        while queue:
            node, depth, idx = queue.pop(0)
            if node is None:
                continue
            if depth <= self.max_depth:
                out.append((node, depth, idx))
    
            if isinstance(node, InnerNode):
                queue.append((node.left, depth + 1, idx * 2))
                queue.append((node.right, depth + 1, idx * 2 + 1))
        return out

    def _batch_decode_and_resize(self, nodes_list: List[Tuple[Any, int, int]]) -> None:
        inner_nodes = [
            (node, depth, idx) 
            for node, depth, idx in nodes_list 
            if isinstance(node, InnerNode)
            ]
        
        if not inner_nodes:
            return
        
        node_size = max(self.layout.width, self.layout.height)
        img_size = min(node_size, max(16, min(256, self.config.img_size)))
        
        wx_list = [
            node.fc.weight.detach().cpu().numpy().flatten() \
                * node.beta.detach().cpu().item() \
                * self.embed
                for node, _, _ in inner_nodes
            ]
        
        wx_array = np.array(wx_list, dtype=np.float32)
        wx_state = State(wx_array, device=self.world_model.device)

        decoded_obs = self.world_model.decode(wx_state)

        for (node, _, _), img in zip(inner_nodes, decoded_obs.for_render):
            self.decoded_cache[node] = self.resize_image(image=img, width=img_size, height=img_size)


    def _draw_all_edges(self, nodes_list: List[Tuple[Any, int, int]]) -> None:
        cumulative_probs = {}
        cumulative_probs[self.tree.root] = 1.0
        leaf_probs = {}

        for node, depth, idx in nodes_list:
            if isinstance(node, InnerNode):
                x, y = self._get_node_position(depth, idx)
                lx, ly = self._get_node_position(depth + 1, idx * 2)
                rx, ry = self._get_node_position(depth + 1, idx * 2 + 1)
                
                device = next(node.fc.parameters()).device
                embed_tensor = torch.tensor(self.embed, dtype=torch.float32, device=device).unsqueeze(0)
                prob_right = node.forward(embed_tensor).item()
                prob_left = 1 - prob_right

                parent_prob = cumulative_probs.get(node, 1.0)
                cumulative_probs[node.left] = parent_prob * prob_left
                cumulative_probs[node.right] = parent_prob * prob_right
                
                self._draw_edge(x, y, lx, ly, cumulative_probs[node.left])
                self._draw_edge(x, y, rx, ry, cumulative_probs[node.right])
            elif isinstance(node, LeafNode):
                leaf_probs[node] = cumulative_probs.get(node, 0.0)
            
            if leaf_probs:
                self.most_probable_leaf = max(leaf_probs, key=leaf_probs.get)
            else:
                self.most_probable_leaf = None


    def _draw_inner_node(self, node: Any, depth: int, idx: int) -> None:
        x, y = self._get_node_position(depth, idx)
        if isinstance(node, LeafNode):
            self._draw_leaf(node, x, y, self.layout.width, self.layout.height)
        else:
            if node in self.decoded_cache:
                img = self.decoded_cache[node]
            else:
                img = self._build_embed_image(node, x, y)

            img_h, img_w = img.shape[:2]
            canvas_h, canvas_w = self.canvas.shape[:2]

            top = max(0, min(canvas_h - img_h, y - img_h // 2))
            left = max(0, min(canvas_w - img_w, x - img_w // 2))
            bottom = min(canvas_h, top + img_h)
            right = min(canvas_w, left + img_w)

            if bottom - top != img_h or right - left != img_w:
                img = img[:bottom-top, :right-left]

            try:
                self.canvas[top:bottom, left:right] = img
            except ValueError as e:
                self.logger.warning(f"Error drawing node at ({x}, {y}): {e}")


    def _resize_image(self, img: np.ndarray, width: int, height: int) -> np.ndarray:
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    def _get_node_position(self, depth: int, node_index: int) -> Tuple[int, int]:
        if depth >= len(self.layout.x_positions):
            self.logger.warning(f"Depth {depth} out of range, clamping.")
            depth = len(self.layout.x_positions) - 1

        if not self.layout.x_positions[depth]:
            self.logger.warning(f"No x_positions at depth={depth}. Returning center screen.")
            return (self.config.window_width // 2, self.config.window_height // 2)

        if node_index >= len(self.layout.x_positions[depth]):
            self.logger.warning(f"Index {node_index} out of range at depth {depth}, clamping.")
            node_index = len(self.layout.x_positions[depth]) - 1

        x = self.layout.x_positions[depth][node_index]

        usable_height = self.config.window_height - 2 * self.config.top_margin
        level_height = usable_height / (self.max_depth + 1)
        y = int(self.config.top_margin + depth * level_height + (level_height / 2))
        return (x, y)


    def _draw_input(
            self,
            embed: np.ndarray,
            x_left: int,
            y_top: int,
            ) -> None:
        """
        Draws a preview of the embedding, from white->red like leaf bars.
        """
        label = "Input Embedding"
        cv2.putText(
            self.canvas,
            label,
            (x_left, y_top - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.font_scale,
            (0, 0, 0),
            self.config.font_thickness
        )
        embed = embed[0]
        box_size = max(50, min(self.config.window_width, self.config.window_height) // 6)

        length = len(embed)
        side = int(math.sqrt(length))
        if side * side != length:
            side = length

        cell_size = max(1, box_size // max(side, 1))
        max_val = np.max(np.abs(embed)) + 1e-8

        for i in range(length):
            val = embed[i]
            alpha = min(1.0, max(0.0, abs(val) / max_val))
            blue = np.array(self.config.blue, dtype=np.float32)
            color = self.blend(np.array(alpha), blue)

            r, c = divmod(i, side)
            sx = x_left + c * cell_size
            sy = y_top + r * cell_size

            cv2.rectangle(
                self.canvas,
                (sx, sy),
                (sx + cell_size, sy + cell_size),
                tuple(int(k) for k in color),
                -1
            )

        w_box = side * cell_size
        h_box = side * cell_size
        cv2.rectangle(
            self.canvas,
            (x_left - 1, y_top - 1),
            (x_left + w_box + 1, y_top + h_box + 1),
            (0, 0, 0),
            1
        )

    def _draw_legend(self,
                     x_left: int,
                     y_top: int) -> None:
        """
        Draws two bars: 
         1) Hue: red->blue
         2) Magnitude: white->red
        With short labels above each bar.
        """
        legend_label = "Legend"
        cv2.putText(self.canvas,
                    legend_label,
                    (x_left, y_top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.config.font_scale,
                    (0, 0, 0),
                    self.config.font_thickness)

        bar_w = max(60, self.config.window_width // 15)
        bar_h = max(6, self.config.window_height // 100)
        gap = bar_h * 2

        # Hue label
        hue_label = "Hue: red(-), blue(+)"
        hue_label_x = x_left
        hue_label_y = y_top + bar_h
        cv2.putText(
            self.canvas,
            hue_label,
            (hue_label_x, hue_label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.font_scale * 0.8,
            (0, 0, 0),
            self.config.font_thickness
        )

        # Hue bar below label
        hue_bar_top = hue_label_y + 8
        steps = 30
        for i in range(steps):
            frac = i / (steps - 1)
            # Red=(0,0,255), Blue=(255,0,0)
            red = np.array(self.config.red, dtype=np.float32)
            blue = np.array(self.config.blue, dtype=np.float32)
            color = self.blend(frac, blue, red)

            xx1 = x_left + i * (bar_w // steps)
            yy1 = hue_bar_top
            xx2 = xx1 + (bar_w // steps)
            yy2 = yy1 + bar_h

            cv2.rectangle(
                self.canvas,
                (xx1, yy1),
                (xx2, yy2),
                tuple(int(k) for k in color),
                -1
            )

        # Next label for magnitude
        mag_label = "Magnitude : 0->1"
        mag_label_x = x_left
        mag_label_y = hue_bar_top + bar_h + gap
        cv2.putText(
            self.canvas,
            mag_label,
            (mag_label_x, mag_label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.font_scale * 0.8,
            (0, 0, 0),
            self.config.font_thickness
        )

        # Magnitude bar
        mag_bar_top = mag_label_y + 8
        steps2 = 30
        for i in range(steps2):
            frac = i / (steps2 - 1)
            blue = np.array(self.config.blue, dtype=np.float32)
            color = self.blend(frac, blue)

            xx1 = x_left + i * (bar_w // steps2)
            yy1 = mag_bar_top
            xx2 = xx1 + (bar_w // steps2)
            yy2 = yy1 + bar_h

            cv2.rectangle(
                self.canvas,
                (xx1, yy1),
                (xx2, yy2),
                tuple(int(k) for k in color),
                -1
            )

    def _draw_edge(self,
                   x1: int,
                   y1: int,
                   x2: int,
                   y2: int,
                   prob: float) -> None:
        """Draw blue edge with thickness reflecting probability values."""

        color = self.config.blue # aways blue
        thickness = max(1, min(5, int(1 + prob * 3)))
        cv2.line(self.canvas, (x1, y1), (x2, y2), color, thickness)

        txt = f"{prob:.2f}"[1:]
        mx = (x1 + x2) // 2
        my = (y1 + y2) // 2 - 10
        (tw, th), _ = cv2.getTextSize(txt,
                                      cv2.FONT_HERSHEY_SIMPLEX,
                                      self.config.font_scale,
                                      self.config.font_thickness)
        pad = 2
        bg1 = (mx - tw // 2 - pad, my - th - pad)
        bg2 = (mx + tw // 2 + pad, my + pad)
        cv2.rectangle(self.canvas, bg1, bg2, (255, 255, 255), -1)
        cv2.putText(self.canvas, txt,
                    (mx - tw // 2, my),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.config.font_scale,
                    (0, 0, 0),
                    self.config.font_thickness)

    def _draw_nodes(self,
                    node: Any,
                    embed: np.ndarray,
                    depth: int,
                    node_index: int) -> None:
        if node is None:
            return

        x, y = self._get_node_position(depth, node_index)

        if isinstance(node, LeafNode):
            self._draw_leaf(node, x, y,
                            self.layout.width,
                            self.layout.height)
        else:
            size = self.layout.level_widths[depth]
            self._draw_inner_node(node, embed, x, y, size)
            self._draw_nodes(node.left, embed, depth + 1, node_index * 2)
            self._draw_nodes(node.right, embed, depth + 1, node_index * 2 + 1)


    def _gather_wx(self, root: InnerNode, embed: np.ndarray) -> Dict[str, np.ndarray]:
        device = next(root.fc.parameters()).device
        wx_map = {}

        def recurse(node: Any) -> None:
            if node is None or isinstance(node, LeafNode):
                return

            if isinstance(node, InnerNode):
                w_raw = node.fc.weight.detach().cpu().numpy().flatten()
                beta_val = node.beta.detach().cpu().item()
                weights = w_raw * beta_val
                wx_values = weights * embed
                wx_map[node] = wx_values
                recurse(node.left)
                recurse(node.right)

        recurse(root)
        return wx_map

    def _draw_leaf(
            self,
            node: LeafNode,
            cx: int,
            cy: int,
            width: int,
            height: int
        ) -> None:
        """Visualizes the leaf node with labeled action probabilities."""

        action_probs = torch.softmax(node.param, dim=0).detach().cpu().numpy()
        label_names = ["forward", "left", "right"]
        n_actions = len(action_probs)

        w_box = width
        h_box = max(1, height // n_actions)
        tl_x = cx - w_box // 2
        tl_y = cy - (h_box * n_actions // 2)

        cmap = cv2.COLORMAP_VIRIDIS
        max_prob = max(action_probs)

        for i, prob in enumerate(action_probs):
            alpha = float(prob)
            norm_alpha = int((alpha / max_prob) * 255) # [0..255] colormap
            blue = np.array(self.config.blue, dtype=np.float32)
            color_map = cv2.applyColorMap(np.array([[norm_alpha]], dtype=np.uint8), cmap)
            color = tuple(int(x) for x in color_map[0][0])

            y1 = tl_y + i * h_box
            cv2.rectangle(
                self.canvas, (tl_x, y1), 
                (tl_x + w_box, y1 + h_box),
                tuple(int(k) for k in color), -1
            )
            cv2.rectangle(
                self.canvas, (tl_x, y1), 
                (tl_x + w_box, y1 + h_box),
                (0, 0, 0), 1
            )

            if self.config.show_prob_text:
                prob_text = f"{alpha:.2f}"
                text_x = tl_x + w_box // 2 - 10
                text_y = y1 + h_box // 2 + 5
                cv2.putText(
                    self.canvas, prob_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
                )

            if self.config.show_label:
                label_x = tl_x - 60
                label_y = y1 + h_box // 2 + 5
                cv2.putText(
                    self.canvas, label_names[i], (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
                )


        if self.most_probable_leaf == node:
            # border for most probable leaf
            cv2.rectangle(
                self.canvas, (tl_x - 2, tl_y - 2),
                (tl_x + w_box + 2, tl_y + h_box * n_actions + 2),
                (0, 0, 0), 3
                )

    def _apply_background_gradient(self) -> None:
        """Creates a visually appealing gradient background."""
        rows, cols, _ = self.canvas.shape
        gradient = np.linspace(100, 230, rows, dtype=np.uint8)[:, np.newaxis]
        gradient = np.repeat(gradient, cols, axis=1)

        # Convert to a color image (blueish gradient)
        blue_channel = np.clip(gradient + 25, 0, 255)
        green_channel = np.clip(gradient - 30, 0, 255)
        red_channel = np.clip(gradient - 50, 0, 255)

        self.canvas[:] = cv2.merge([blue_channel, green_channel, red_channel])



if __name__ == "__main__":
    torch.manual_seed(42)
    config = SoftConfig()

    input_dim = 64
    output_dim = 3
    sample_x = torch.randn(1, input_dim)

    tree = SoftDecisionTree(config)
    tree.setup_network(sample_x, torch.randn(1, output_dim))

    from src.run.config import TreeVisualizerConfig
    viz_cfg = TreeVisualizerConfig(
        window_width=1200,
        window_height=900,
        show_embed=True,
        show_legend=True
    )

    visualizer = SoftTreeVisualizer(viz_cfg, tree)
    visualizer.update(sample_x)
    visualizer.render()
