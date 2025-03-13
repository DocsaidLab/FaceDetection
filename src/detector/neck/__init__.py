from chameleon import Registry

from .fpn import FPN
from .pafpn import PAFPN

NECKS = Registry("neck")
NECKS.register_module("mm_fpn", module=FPN)
NECKS.register_module("mm_pafpn", module=PAFPN)
