import numpy as np
import matplotlib.pyplot as plt
import torch

class TreeVisualizer:
    def __init__(self, model, image_shape):
        """
        Initialize the tree visualizer
        model: SoftDecisionTree instance
        image_shape: tuple (H, W, C)
        """
        self.model = model
        self.H, self.W, self.C = image_shape
        self.depth = model.max_depth
        
        # Calculate layout dimensions
        self.levels = self.depth + 1  # including leaf level
        
        # Figure properties
        self.node_width = 4  # inches for internal nodes
        self.node_height = 3  # inches for internal nodes
        self.leaf_height = 1.0  # inches for leaf nodes
        self.horizontal_spacing = 1  # inches
        self.vertical_spacing = 0.5  # inches
        self.leaf_gap = 0.3  # inches (gap before leaf nodes)
        
        # Calculate figure dimensions based on deepest level
        self.max_nodes_last_level = 2 ** self.depth
        self.total_width = self.max_nodes_last_level * self.node_width + \
                          (self.max_nodes_last_level - 1) * self.horizontal_spacing
        self.total_height = ((self.depth - 1) * (self.node_height + self.vertical_spacing)) + \
                           self.node_height + self.leaf_gap + self.leaf_height
    
    def _get_node_weights(self, node):
        """Extract and reshape weights from a node's linear layer"""
        if node.leaf:
            return None
        weights = node.fc.weight.data.cpu().numpy().reshape(-1)
        return weights.reshape(self.H, self.W, self.C)
    
    def _create_figure(self):
        """Create figure with appropriate size for the tree"""
        fig = plt.figure(figsize=(self.total_width, self.total_height), dpi=150)
        return fig
    
    def _get_node_position(self, level, position):
        """Calculate the centered position for a node in the figure"""
        nodes_in_level = 2 ** level
        node_spacing = self.total_width / nodes_in_level
        x_center = (position + 0.5) * node_spacing
        
        # Center the node within its allocated space
        x = (x_center - self.node_width/2) / self.total_width
        
        # Calculate y position, accounting for different heights
        if level == self.depth:  # leaf level
            y = 0
            height = self.leaf_height
        else:
            # Calculate y position for internal nodes, with gap before leaves
            base_height = self.leaf_height + self.leaf_gap
            remaining_height = self.total_height - base_height
            y = base_height + (self.depth - 1 - level) * (remaining_height / (self.depth))
            height = self.node_height
            
        y = y / self.total_height
        
        return x, y, self.node_width/self.total_width, height/self.total_height
    
    def visualize_node(self, ax, node, image, weights=None, path_prob=None):
        """Visualize a single node"""
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        if node.leaf:
            # For leaf nodes, show action probabilities as vertical stacks
            probs = node.forward().detach().cpu().numpy().squeeze()
            for i, prob in enumerate(probs):
                color = plt.cm.Greys(prob)
                ax.add_patch(plt.Rectangle((0, (2-i)/3), 1, 1/3, color=color))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        else:
            # Reshape flat image back to HxWxC
            image_reshaped = image.reshape(self.H, self.W, self.C)
            
            # For inner nodes, show image and weight overlay
            img_bw = image_reshaped.mean(axis=2)
            ax.imshow(img_bw, cmap='gray', aspect='auto')
            
            if weights is not None:
                # Create a color mask from weights
                weight_intensity = np.abs(weights).mean(axis=2)
                weight_intensity = (weight_intensity - weight_intensity.min()) / \
                                 (weight_intensity.max() - weight_intensity.min() + 1e-8)
                colored_weights = plt.cm.viridis(weight_intensity)
                ax.imshow(colored_weights, alpha=0.5, aspect='auto')
        
        # Add path probability indicator
        if path_prob is not None:
            # Add a small rectangle at the top of the node showing cumulative probability
            rect_height = 0.1  # relative to axes height
            rect = plt.Rectangle((0, 1), 1, rect_height, 
                               transform=ax.transAxes, 
                               color='red', 
                               alpha=path_prob,
                               clip_on=False)
            ax.add_patch(rect)
        
        # Remove borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    def visualize_path(self, image, path_probs=None):
        """
        Visualize the decision path for a single image
        image: numpy array of shape (H, W, C)
        path_probs: dict of node path probabilities
        """
        fig = self._create_figure()
        
        def traverse(node, level=0, position=0):
            x, y, w, h = self._get_node_position(level, position)
            ax = fig.add_axes([x, y, w, h])
            
            weights = self._get_node_weights(node)
            path_prob = path_probs.get(node) if path_probs else None
            self.visualize_node(ax, node, image, weights, path_prob)
            
            if not node.leaf:
                traverse(node.left, level + 1, position * 2)
                traverse(node.right, level + 1, position * 2 + 1)
        
        traverse(self.model.root)
        plt.show()
        
    def visualize_decision(self, image):
        """
        Visualize the decision process for a single image, including path probabilities
        image: numpy array of shape (H, W, C)
        """
        # Get path probabilities for this image
        path_probs = self.model.get_path_probabilities(image.reshape(1, -1))
        self.visualize_path(image, path_probs)