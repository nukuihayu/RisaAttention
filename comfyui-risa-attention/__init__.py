"""ComfyUI custom-node entry point for RISA Attention."""

from .node import RISAAttentionNode

NODE_CLASS_MAPPINGS = {"RISAAttention": RISAAttentionNode}
NODE_DISPLAY_NAME_MAPPINGS = {"RISAAttention": "RISA Attention"}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
