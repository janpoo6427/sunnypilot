"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from enum import IntEnum

from openpilot.common.params import Params
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.multilang import tr, tr_noop
from openpilot.system.ui.sunnypilot.widgets.list_view import toggle_item_sp, option_item_sp, simple_button_item_sp
from openpilot.system.ui.widgets.scroller_tici import Scroller
from openpilot.system.ui.widgets import Widget

class PanelType(IntEnum):
  CRUISE = 0
  SLA = 1

ICBM_DESC = tr_noop("When enabled, sunnypilot will attempt to manage the built-in cruise control buttons by emulating button presses for limited longitudinal control.")
ICMB_UNAVAILABLE = tr_noop("Intelligent Cruise Button Management is currently unavailable on this platform.")
ICMB_UNAVAILABLE_LONG_AVAILABLE = tr_noop("Disable the sunnypilot Longitudinal Control (alpha) toggle to allow Intelligent Cruise Button Management.")
ICMB_UNAVAILABLE_LONG_UNAVAILABLE = tr_noop("sunnypilot Longitudinal Control is the default longitudinal control for this platform.")

ACC_ENABLED_DESCRIPTION = tr_noop("Enable custom Short & Long press increments for cruise speed increase/decrease.")
ACC_NOLONG_DESCRIPTION = tr_noop("This feature can only be used with sunnypilot longitudinal control enabled.")
ACC_PCMCRUISE_DISABLED_DESCRIPTION = tr_noop("This feature is not supported on this platform due to vehicle limitations.")
ONROAD_ONLY_DESCRIPTION = tr_noop("Start the vehicle to check vehicle compatibility.")

class CruiseLayout(Widget):
  def __init__(self):
    super().__init__()

    self._params = Params()
    items = self._initialize_items()
    self._scroller = Scroller(items, line_separator=True, spacing=0)

  def _initialize_items(self):

    self.icbm_toggle = toggle_item_sp(
      title=tr("Intelligent Cruise Button Management (ICBM) (Alpha)"),
      description="",
      param="IntelligentCruiseButtonManagement")

    self.scc_v_toggle = toggle_item_sp(
      title=tr("Smart Cruise Control - Vision"),
      description=tr("Use vision path predictions to estimate the appropriate speed to drive through turns ahead."),
      param="SmartCruiseControlVision")

    self.scc_m_toggle = toggle_item_sp(
      title=tr("Smart Cruise Control - Map"),
      description=tr("Use map data to estimate the appropriate speed to drive through turns ahead."),
      param="SmartCruiseControlMap")

    self.custom_acc_toggle = toggle_item_sp(
      title=tr("Custom ACC Speed Increments"),
      description="",
      param="CustomAccIncrementsEnabled")

    self.custom_acc_short_increment = option_item_sp(
      title=tr("Short Press Increment"),
      param="CustomAccShortPressIncrement",
      min_value=1, max_value=10, value_change_step=1,
      inline=True)

    self.custom_acc_long_increment = option_item_sp(
      title=tr("Long Press Increment"),
      param="CustomAccLongPressIncrement",
      value_map={1: 1, 2: 5, 3: 10},
      min_value=1, max_value=3, value_change_step=1,
      inline=True)

    self.sla_settings_button = simple_button_item_sp(
      button_text=lambda: tr("Speed Limit"),
      button_width=800,
      callback=lambda: self._set_current_panel(PanelType.SLA)
    )

    self.dec_toggle = toggle_item_sp(
      title=tr("Enable Dynamic Experimental Control"),
      description=tr("Enable toggle to allow the model to determine when to use sunnypilot ACC or sunnypilot End to End Longitudinal."),
      param="DynamicExperimentalControl")

    items = [
      self.icbm_toggle,
      self.scc_v_toggle,
      self.scc_m_toggle,
      self.custom_acc_toggle,
      self.custom_acc_short_increment,
      self.custom_acc_long_increment,
      self.sla_settings_button,
      self.dec_toggle
    ]
    return items

  def _render(self, rect):
    if self._current_panel == PanelType.SLA:
      return
    else:
      self._scroller.render(rect)

  def show_event(self):
    self._set_current_panel(PanelType.CRUISE)
    self._scroller.show_event()

  def _set_current_panel(self, panel: PanelType):
    self._current_panel = panel

  def _update_state(self):
    super()._update_state()

    if ui_state.CP is not None and ui_state.CP_SP is not None:
      if ui_state.CP_SP.intelligentCruiseButtonManagementAvailable and not ui_state.has_longitudinal_control:
        self.icbm_toggle.set_enabled(ui_state.is_offroad())
        self.icbm_toggle.set_description(tr(ICBM_DESC))
      else:
        ui_state.params.remove("IntelligentCruiseButtonManagement")
        self.icbm_toggle.set_enabled(False)

