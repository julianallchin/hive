from vmas import render_interactively
from scenerio import Scenario
my_scenario_instance = Scenario()

render_interactively(
    scenario=my_scenario_instance,
    control_two_agents=True,
    save_render=False,
    display_info=True,
)