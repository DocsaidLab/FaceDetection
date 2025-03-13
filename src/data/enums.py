from enum import Enum

import capybara as cb


class FacePose(cb.EnumCheckMixin, Enum):
    LeftProfile = 0
    LeftFrontal = 1
    Frontal = 2
    RightFrontal = 3
    RightProfile = 4
    UpFrontal = 5
    DownFrontal = 6
    Unknown = -1
