import chameleon as cl

from .scrfd_head import SCRFDHead

HEADS = cl.Registry("head")

HEADS.register_module("scrfd_head", module=SCRFDHead)
