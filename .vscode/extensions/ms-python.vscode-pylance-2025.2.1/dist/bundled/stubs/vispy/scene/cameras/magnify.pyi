import numpy as np

from ...util.event import Event
from ...visuals.transforms.nonlinear import Magnify1DTransform, MagnifyTransform
from .panzoom import PanZoomCamera

# -*- coding: utf-8 -*-
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

class MagnifyCamera(PanZoomCamera):
    transform_class = ...

    def __init__(self, size_factor: float = 0.25, radius_ratio: float = 0.9, **kwargs): ...
    def _viewbox_set(self, viewbox): ...
    def _viewbox_unset(self, viewbox): ...
    def viewbox_mouse_event(self, event: Event): ...
    def on_timer(self, event: Event | None = None): ...
    def viewbox_resize_event(self, event: Event): ...
    def view_changed(self): ...

class Magnify1DCamera(MagnifyCamera):
    transform_class = ...
    __doc__ = ...
