import cv2
import numpy as np
import torch
import math
from dataclasses import dataclass
from typing import Any, List, Tuple, Optional, Dict

from src.run.logger import create_logger
from src.run.config import TreeVisualizerConfig, SoftConfig
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
    Visualizes a differentiable decision tree, with a bottom histogram
    showing the cumulative probability of ending up in each leaf.
    """
    def __init__(self, config: TreeVisualizerConfig) -> None:
        super().__init__(config)
        self.logger = create_logger("tree_viz.log")
        self.config = config
        self.hist_height = int(self.config.window_height * 0.12) + 10
        self.canvas = np.full(
            (config.window_height, config.window_width, 3),
            config.bgc,
            dtype=np.uint8
        )
        self.tree: Optional[SoftDecisionTree] = None
        self.world_model: Optional[WorldModel] = None
        self.decoded_cache: Dict[InnerNode, np.ndarray] = {}
        self.embed = np.zeros((1, config.embedding_width))
        self.labeled: bool = False
        self.leaf_probs: Dict[LeafNode, float] = {}

    def update(
        self,
        embed: np.ndarray,
        tree: Optional[SoftDecisionTree] = None,
        world_model: Optional[WorldModel] = None
    ) -> None:
        """
        Updates the tree visualization dynamically.
        """
        self.labeled = False
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

        if hasattr(tree, 'metric') and tree.metric is not None:
            self._draw_metric()

        # Draw the leaf-probability histogram at the bottom
        self._draw_leaf_prob_histogram()

    def _initialize_layout(self, tree: SoftDecisionTree):
        if tree.root is None:
            return
        self.tree = tree
        self.max_depth = self.tree.max_depth
        self.level_counts = self._compute_level_counts()
        self.layout = self._compute_layout()

    def _compute_level_counts(self) -> List[int]:
        counts = [0] * (self.max_depth + 2)
        def recurse(node: Any, depth: int):
            if node is None:
                return
            counts[depth] += 1
            if isinstance(node, InnerNode):
                recurse(node.left, depth+1)
                recurse(node.right, depth+1)
        recurse(self.tree.root, 0)
        return counts

    def _compute_layout(self) -> NodeLayout:
        usable_w = self.config.window_width - 2 * self.config.lateral_margin
        usable_h = (self.config.window_height
                    - 2 * self.config.top_margin
                    - self.hist_height)
        max_nodes = max(self.level_counts)
        node_w = usable_w // max(max_nodes, 1)
        level_h = usable_h // max(self.max_depth+1, 1)
        node_h = min(int(level_h * 0.6), node_w)

        level_widths = [
            int(node_w * (1 + (self.max_depth - lvl) * 0.15))
            for lvl in range(self.max_depth+1)
        ]

        x_positions = []
        for lvl, count in enumerate(self.level_counts):
            if count <= 1:
                x_positions.append([self.config.window_width//2] if count==1 else [])
            else:
                spacing = usable_w / (count+1)
                x_positions.append([
                    int(self.config.lateral_margin + (i+1)*spacing)
                    for i in range(count)
                ])

        return NodeLayout(width=int(node_w*0.8),
                          height=node_h,
                          x_positions=x_positions,
                          level_widths=level_widths)

    def _collect_nodes_bfs(self, root: Any) -> List[Tuple[Any,int,int]]:
        queue = [(root,0,0)]
        out = []
        while queue:
            node, d, idx = queue.pop(0)
            if node is None: continue
            if d <= self.max_depth:
                out.append((node,d,idx))
            if isinstance(node, InnerNode):
                queue.append((node.left,  d+1, idx*2))
                queue.append((node.right, d+1, idx*2+1))
        return out

    def _batch_decode_and_resize(self, nodes_list):
        inner = [(n,d,i) for n,d,i in nodes_list if isinstance(n,InnerNode)]
        if not inner: return
        img_sz = min(self.layout.width, self.layout.height)

        wxs = []
        nodes = []

        p_left = 0.95
        p_right = 1.0 - p_left
        logit_left   = np.log(p_left / p_right)
        logit_right  = np.log(p_right / p_left)

        for node,_,_ in inner:
            beta   = float(node.beta.detach().cpu().item())
            w_arr  = beta * node.fc.weight.detach().cpu().numpy().flatten()
            b_val  = beta * float(node.fc.bias.detach().cpu().item())
            w_norm2 = np.dot(w_arr, w_arr) + 1e-12

            # compute step so that w^T (step w_unit) + b = logit_*
            alpha_left  = (logit_left - b_val) / w_norm2
            alpha_right = (logit_right - b_val) / w_norm2

            sep = self.config.separation_factor
            alpha_left  *= sep
            alpha_right *= sep

            v_left = alpha_left * w_arr
            v_right = alpha_right * w_arr

            if hasattr(self.tree, "x_mean") and hasattr(self.tree, "x_std"):
                mean = self.tree.x_mean.detach().cpu().numpy()
                std  = self.tree.x_std.detach().cpu().numpy()
                
                v_left  = v_left  * std + mean
                v_right = v_right * std + mean

            wxs.append(v_left)
            wxs.append(v_right)
            nodes.append(node)

        wx_tensor = torch.tensor(np.stack(wxs), dtype=torch.float32, device=self.world_model.device)
        decoded = self.world_model.decode(State(wx_tensor.cpu().numpy(),device=self.world_model.device))
        
        for idx, node in enumerate(nodes):
            img_L = decoded.for_render[2*idx]
            img_R = decoded.for_render[2*idx + 1]

            small_sz = 60
            height_sz = int(small_sz*4/3)
            small_L = cv2.resize(img_L,  (small_sz, height_sz), interpolation=cv2.INTER_AREA)
            small_R = cv2.resize(img_R,  (small_sz, height_sz), interpolation=cv2.INTER_AREA)

            border_color = (0, 0, 0)
            border_width = 2
            border = np.full((height_sz, border_width, 3), border_color, dtype=np.uint8)

            # composite left|right
            composite = np.hstack([small_L, border, small_R])
            self.decoded_cache[node] = composite

    def _draw_all_edges(self, nodes_list):
        cum = {self.tree.root:1.0}
        leaf_probs = {}
        for node,d,idx in nodes_list:
            if isinstance(node,InnerNode):
                x0,y0 = self._get_node_position(d,idx)
                x1,y1 = self._get_node_position(d+1,idx*2)
                x2,y2 = self._get_node_position(d+1,idx*2+1)
                emb = torch.tensor(self.embed,dtype=torch.float32,
                                   device=next(node.fc.parameters()).device).unsqueeze(0)
                pL = float(node.forward(emb).item()); pR = 1-pL
                parent_p = cum[node]
                cum[node.left]  = parent_p * pL
                cum[node.right] = parent_p * pR
                self._draw_edge(x0,y0,x1,y1,pL)
                self._draw_edge(x0,y0,x2,y2,pR)
            elif isinstance(node,LeafNode):
                leaf_probs[node] = cum.get(node,0.0)
        self.most_probable_leaf = max(leaf_probs, key=leaf_probs.get) if leaf_probs else None
        self.leaf_probs = leaf_probs

    def _draw_leaf_prob_histogram(self):
        if not self.leaf_probs or not self.tree: return
        nodes = self._collect_nodes_bfs(self.tree.root)
        leaves = [(idx,self.leaf_probs.get(n,0.0))
                  for n,depth,idx in nodes
                  if isinstance(n,LeafNode) and depth==self.max_depth]
        leaves.sort(key=lambda x:x[0])
        if not leaves: return
        xs = self.layout.x_positions[self.max_depth]
        if not xs: return

        bottom = self.config.window_height - 5
        top_bg = bottom - self.hist_height + 5
        cv2.rectangle(
            self.canvas, 
            (0,top_bg),
            (self.config.window_width, bottom),
            self.config.bgc, -1
            )
        bar_w = int(((xs[1]-xs[0]) if len(xs)>1 else self.layout.width)*0.6)

        for (idx,prob),x in zip(leaves,xs):
            h = int(prob*self.hist_height)
            y0 = bottom - h
            x0, x1 = x-bar_w//2, x+bar_w//2
            cv2.rectangle(
                self.canvas, (x0,y0),
                (x1,bottom), self.config.blue,
                -1
                )
            cv2.putText(self.canvas, f"{prob:.2f}",
                        (x0,y0-4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.config.font_scale*0.8,
                        (0,0,0),
                        self.config.font_thickness)

    def _get_node_position(self, depth: int, idx: int) -> Tuple[int, int]:
        """
        Return (x,y) for a node at given depth/idx, pushing all nodes
        above the reserved histogram strip at the bottom.
        """
        # clamp depth
        if depth >= len(self.layout.x_positions):
            depth = len(self.layout.x_positions) - 1

        xs = self.layout.x_positions[depth]
        if not xs:
            # fallback to center
            return (self.config.window_width // 2,
                    self.config.window_height // 2 - self.hist_height // 2)

        # clamp index
        if idx >= len(xs):
            idx = len(xs) - 1
        x = xs[idx]

        # compute vertical spacing within the content area (excluding histogram)
        content_h = (self.config.window_height
                     - 2 * self.config.top_margin
                     - self.hist_height)
        level_h = content_h / max(self.max_depth + 1, 1)
        y = int(self.config.top_margin + depth * level_h + level_h / 2)

        return (x, y)

    def _build_embed_image(self, node: InnerNode, cx:int, cy:int) -> np.ndarray:
        """Fallback image when no world_model decode is available."""
        w, h = self.layout.width, self.layout.height
        img = np.full((h,w,3), self.config.bgc, dtype=np.uint8)
        cv2.rectangle(img,(0,0),(w-1,h-1),(0,0,0),1)
        return img

    def _draw_inner_node(self, node: Any, depth:int, idx:int) -> None:
        x,y = self._get_node_position(depth,idx)
        if isinstance(node,LeafNode):
            self._draw_leaf(node,x,y,self.layout.width,self.layout.height)
        else:
            img = self.decoded_cache.get(node,
                                        self._build_embed_image(node,x,y))
            h,w = img.shape[:2]
            ch,cw = self.canvas.shape[:2]
            top  = max(0, min(ch-h, y-h//2))
            left = max(0, min(cw-w, x-w//2))
            self.canvas[top:top+h,left:left+w] = img

    def _draw_edge(self, x1:int,y1:int,x2:int,y2:int,prob:float) -> None:
        thickness = max(1, min(5,int(1 + prob*3)))
        cv2.line(self.canvas,(x1,y1),(x2,y2),self.config.blue,thickness)
        mx,my = (x1+x2)//2, (y1+y2)//2 - 10
        txt = f"{prob:.2f}"[1:]
        (tw,th),_ = cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,
                                    self.config.font_scale,
                                    self.config.font_thickness)
        cv2.rectangle(self.canvas,
                      (mx-tw//2-2,my-th-2),
                      (mx+tw//2+2,my+2),
                      (255,255,255),-1)
        cv2.putText(self.canvas,txt,
                    (mx-tw//2,my),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.config.font_scale,
                    (0,0,0),
                    self.config.font_thickness)

    def _draw_input(self, embed: np.ndarray, x_left:int, y_top:int) -> None:
        label = "Input Embedding"
        cv2.putText(self.canvas,label,(x_left,y_top-8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.config.font_scale,(0,0,0),self.config.font_thickness)
        arr = embed[0]
        box = max(50,min(self.config.window_width,self.config.window_height)//6)
        L = len(arr)
        side = int(math.sqrt(L))
        if side*side!=L: side=L
        cell = max(1,box//max(side,1))
        maxv = np.max(np.abs(arr))+1e-8
        for i,val in enumerate(arr):
            alpha = min(1.0,max(0.0,abs(val)/maxv))
            color = self.blend(alpha,np.array(self.config.blue,dtype=np.float32))
            r,c = divmod(i,side)
            sx,sy = x_left+c*cell, y_top+r*cell
            cv2.rectangle(self.canvas,(sx,sy),(sx+cell,sy+cell),
                          tuple(int(k) for k in color),-1)
        w_box, h_box = side*cell, side*cell
        cv2.rectangle(self.canvas,
                      (x_left-1,y_top-1),
                      (x_left+w_box+1,y_top+h_box+1),
                      (0,0,0),1)

    def _draw_legend(self, x_left:int, y_top:int) -> None:
        cv2.putText(self.canvas,"Legend",(x_left,y_top-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.config.font_scale,(0,0,0),self.config.font_thickness)
        bar_w = max(60,self.config.window_width//15)
        bar_h = max(6,self.config.window_height//100)
        gap = bar_h*2
        # Hue
        cv2.putText(self.canvas,"Hue: red(-), blue(+)",
                    (x_left,y_top+bar_h),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.config.font_scale*0.8,(0,0,0),self.config.font_thickness)
        top_h = y_top+bar_h+8
        for i in range(30):
            frac = i/29
            color = self.blend(frac,
                               np.array(self.config.blue,dtype=np.float32),
                               np.array(self.config.red, dtype=np.float32))
            xx1 = x_left + i*(bar_w//30)
            cv2.rectangle(self.canvas,(xx1,top_h),
                          (xx1+bar_w//30,top_h+bar_h),
                          tuple(int(k) for k in color),-1)
        # Magnitude
        mag_y = top_h+bar_h+gap
        cv2.putText(self.canvas,"Magnitude : 0->1",
                    (x_left,mag_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.config.font_scale*0.8,(0,0,0),self.config.font_thickness)
        top_m = mag_y+8
        for i in range(30):
            frac = i/29
            color = self.blend(frac,np.array(self.config.blue,dtype=np.float32))
            xx1 = x_left + i*(bar_w//30)
            cv2.rectangle(self.canvas,(xx1,top_m),
                          (xx1+bar_w//30,top_m+bar_h),
                          tuple(int(k) for k in color),-1)

    def _draw_leaf(self, node:LeafNode, cx:int, cy:int, w:int, h:int) -> None:
        probs = torch.softmax(node.param,dim=0).cpu().numpy()
        names = ["forward","left","right"]
        n = len(probs)
        h_box = max(1,h//n)
        tlx = cx - w//2
        tly = cy - (h_box*n//2)
        cmap = cv2.COLORMAP_VIRIDIS
        m = max(probs)
        for i,p in enumerate(probs):
            alpha = int((p/m)*255)
            cmap_col = cv2.applyColorMap(np.array([[alpha]],dtype=np.uint8),cmap)[0][0]
            y1 = tly + i*h_box
            cv2.rectangle(self.canvas,(tlx,y1),(tlx+w,y1+h_box),
                          tuple(int(c) for c in cmap_col),-1)
            cv2.rectangle(self.canvas,(tlx,y1),(tlx+w,y1+h_box),(0,0,0),1)
            if self.config.show_prob_text:
                txt = f"{p:.2f}"
                tx = tlx + w//2 - 10
                ty = y1 + h_box//2 + 5
                cv2.putText(self.canvas,txt,(tx,ty),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
            if self.config.show_label and not self.labeled:
                lx = tlx - 60
                ly = y1 + h_box//2 +5
                cv2.putText(self.canvas,names[i],(lx,ly),
                            cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,0),1)
        self.labeled = True
        if node is self.most_probable_leaf:
            cv2.rectangle(self.canvas,
                          (tlx-2,tly-2),
                          (tlx+w+2,tly+h_box*n+2),
                          (0,0,0),3)

    def _draw_metric(self) -> None:
        margin = 20; bw=200; bh=32
        fs=0.65; ft=2
        x = self.config.window_width - bw - margin; y=margin
        cv2.rectangle(self.canvas,(x,y),(x+bw,y+bh),(240,240,240),-1)
        cv2.rectangle(self.canvas,(x,y),(x+bw,y+bh),(0,0,0),2)
        txt = f"Argmax Acc: {self.tree.metric:.2f}"
        (tw,th),_ = cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,fs,ft)
        tx = x + (bw-tw)//2; ty = y + (bh+th)//2 -5
        cv2.putText(self.canvas,txt,(tx,ty),
                    cv2.FONT_HERSHEY_SIMPLEX,fs,(0,0,0),ft,cv2.LINE_AA)

    def _apply_background_gradient(self) -> None:
        rows,cols,_ = self.canvas.shape
        grad = np.linspace(100,230,rows,dtype=np.uint8)[:,None]
        grad = np.repeat(grad,cols,axis=1)
        b = np.clip(grad+25,0,255)
        g = np.clip(grad-30,0,255)
        r = np.clip(grad-50,0,255)
        self.canvas[:] = cv2.merge([b,g,r])


if __name__ == "__main__":
    torch.manual_seed(42)
    cfg_tree = TreeVisualizerConfig(
        window_width=1200,
        window_height=900,
        show_embed=True,
        show_legend=True
    )
    # build a dummy tree
    cfg_soft = SoftConfig()
    tree = SoftDecisionTree(cfg_soft)
    x = torch.randn(1, cfg_soft.latent_dim)
    y = torch.randn(1, cfg_soft.lmbda.shape[0] if hasattr(cfg_soft,'lmbda') else 3)
    tree.setup_network(x, torch.randn(1,3))
    viz = SoftTreeVisualizer(cfg_tree)
    viz.update(x.cpu().numpy(), tree, None)
    cv2.imshow("Tree", viz.canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
