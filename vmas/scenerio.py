#  Copyright (c) 2022-2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import typing # Import typing for type hints
from typing import Callable # Import Callable for type hints

import torch

from vmas import render_interactively
from vmas.simulator.core import Agent, Box, Entity, Landmark, Sphere, World # Added Entity for filter
from vmas.simulator.heuristic_policy import BaseHeuristicPolicy
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.sensors import Lidar # Import Lidar
from vmas.simulator.utils import Color, ScenarioUtils


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        n_agents = kwargs.pop("n_agents", 4)
        self.n_packages = kwargs.pop("n_packages", 1)
        self.package_width = kwargs.pop("package_width", 0.15)
        self.package_length = kwargs.pop("package_length", 0.15)
        self.package_mass = kwargs.pop("package_mass", 50)

        # New parameters for LIDAR and obstacles
        self.n_obstacles = kwargs.pop("n_obstacles", 3)
        self.obstacle_radius = kwargs.pop("obstacle_radius", 0.1)
        self._lidar_range = kwargs.pop("lidar_range", 0.5)
        self.n_lidar_rays_agents = kwargs.pop("n_lidar_rays_agents", 16)
        self.n_lidar_rays_packages = kwargs.pop("n_lidar_rays_packages", 16)
        self.n_lidar_rays_obstacles = kwargs.pop("n_lidar_rays_obstacles", 16)

        ScenarioUtils.check_kwargs_consumed(kwargs)

        self.shaping_factor = 100
        self.world_semidim = 1
        self.agent_radius = 0.03

        # Make world
        world = World(
            batch_dim,
            device,
            x_semidim=self.world_semidim
            + 2 * self.agent_radius
            + max(self.package_length, self.package_width),
            y_semidim=self.world_semidim
            + 2 * self.agent_radius
            + max(self.package_length, self.package_width),
        )

        # LIDAR entity filters
        # We need to be careful with lambda functions if they capture `self` or other
        # variables that might change, but for simple name checks or type checks it's usually fine.
        # To be safer, we can define them as local functions if they depend on instance attributes
        # that are set up before agent creation.
        # However, for this case, direct type checking or name checking is simple enough.

        entity_filter_agents: Callable[[Entity], bool] = (
            lambda e: isinstance(e, Agent) and e.name.startswith("agent")
        )
        # Filter for packages (assuming their names start with "package")
        entity_filter_packages: Callable[[Entity], bool] = (
            lambda e: isinstance(e, Landmark) and e.name.startswith("package")
        )
        # Filter for obstacles (assuming their names start with "obstacle")
        entity_filter_obstacles: Callable[[Entity], bool] = (
            lambda e: isinstance(e, Landmark) and e.name.startswith("obstacle")
        )

        # Add agents
        for i in range(n_agents):
            agent_sensors = [
                Lidar( # Lidar for other agents
                    world,
                    n_rays=self.n_lidar_rays_agents,
                    max_range=self._lidar_range,
                    entity_filter=entity_filter_agents,
                    render_color=Color.BLUE, # Different color for visualization
                ),
                Lidar( # Lidar for packages
                    world,
                    n_rays=self.n_lidar_rays_packages,
                    max_range=self._lidar_range,
                    entity_filter=entity_filter_packages,
                    render_color=Color.ORANGE, # Different color
                ),
                Lidar( # Lidar for obstacles
                    world,
                    n_rays=self.n_lidar_rays_obstacles,
                    max_range=self._lidar_range,
                    entity_filter=entity_filter_obstacles,
                    render_color=Color.BLACK, # Different color
                ),
            ]
            agent = Agent(
                name=f"agent_{i}",
                shape=Sphere(self.agent_radius),
                u_multiplier=0.6,
                sensors=agent_sensors, # Add sensors to the agent
            )
            world.add_agent(agent)

        # Add landmarks
        goal = Landmark(
            name="goal",
            collide=False,
            shape=Sphere(radius=0.15),
            color=Color.LIGHT_GREEN,
        )
        world.add_landmark(goal)

        self.packages = []
        for i in range(self.n_packages):
            package = Landmark(
                name=f"package {i}", # Ensure name starts with "package "
                collide=True,
                movable=True,
                mass=self.package_mass,
                shape=Box(length=self.package_length, width=self.package_width),
                color=Color.RED,
            )
            package.goal = goal
            self.packages.append(package)
            world.add_landmark(package)

        # Add obstacles
        self.obstacles = []
        for i in range(self.n_obstacles):
            obstacle = Landmark(
                name=f"obstacle_{i}", # Ensure name starts with "obstacle"
                collide=True, # Obstacles should be collidable
                movable=False, # Obstacles are static
                shape=Sphere(radius=self.obstacle_radius),
                color=Color.GRAY, # Distinguish from packages/goal
            )
            self.obstacles.append(obstacle)
            world.add_landmark(obstacle)


        return world

    def reset_world_at(self, env_index: int = None):
        # Random pos for agents
        ScenarioUtils.spawn_entities_randomly(
            self.world.agents,
            self.world,
            env_index,
            min_dist_between_entities=self.agent_radius * 2.1, # ensure no overlap
            x_bounds=(-self.world_semidim, self.world_semidim),
            y_bounds=(-self.world_semidim, self.world_semidim),
        )
        agent_occupied_positions_list = [agent.state.pos for agent in self.world.agents]
        # If env_index is None, stack all, else take the specific env_index
        if env_index is None:
            agent_occupied_positions = torch.stack(agent_occupied_positions_list, dim=1) if agent_occupied_positions_list else torch.empty((self.world.batch_dim, 0, 2), device=self.world.device)
        else:
            agent_occupied_positions = torch.stack([pos[env_index] for pos in agent_occupied_positions_list], dim=0).unsqueeze(0) if agent_occupied_positions_list else torch.empty((1, 0, 2), device=self.world.device)


        # Random pos for goal and packages, avoiding agent positions
        goal = self.world.landmarks[0] # Assuming goal is always the first landmark added
        entities_to_spawn_after_agents = [goal] + self.packages
        min_dist_goal_package = max(
            (p.shape.circumscribed_radius() + goal.shape.radius + 0.01 for p in self.packages), default=0.01
        )
        ScenarioUtils.spawn_entities_randomly(
            entities_to_spawn_after_agents,
            self.world,
            env_index,
            min_dist_between_entities=min_dist_goal_package,
            x_bounds=(-self.world_semidim, self.world_semidim),
            y_bounds=(-self.world_semidim, self.world_semidim),
            occupied_positions=agent_occupied_positions,
        )

        # Collect positions of agents, goal, and packages for obstacle spawning
        current_occupied_positions_list = agent_occupied_positions_list + \
                                     [goal.state.pos] + \
                                     [p.state.pos for p in self.packages]
        if env_index is None:
            all_occupied_positions = torch.stack(current_occupied_positions_list, dim=1) if current_occupied_positions_list else torch.empty((self.world.batch_dim, 0, 2), device=self.world.device)
        else:
            all_occupied_positions = torch.stack([pos[env_index] for pos in current_occupied_positions_list], dim=0).unsqueeze(0) if current_occupied_positions_list else torch.empty((1,0,2), device=self.world.device)


        # Random pos for obstacles, avoiding agents, goal, and packages
        ScenarioUtils.spawn_entities_randomly(
            self.obstacles,
            self.world,
            env_index,
            min_dist_between_entities=self.obstacle_radius * 2.1 + self.agent_radius, # Min dist from other entities
            x_bounds=(-self.world_semidim, self.world_semidim),
            y_bounds=(-self.world_semidim, self.world_semidim),
            occupied_positions=all_occupied_positions,
        )


        for package in self.packages:
            # Check overlap with goal after placement
            if env_index is None:
                package.on_goal = self.world.is_overlapping(package, package.goal)
                package.global_shaping = (
                    torch.linalg.vector_norm(
                        package.state.pos - package.goal.state.pos, dim=1
                    )
                    * self.shaping_factor
                )
            else:
                package.on_goal[env_index] = self.world.is_overlapping(
                    package, package.goal, env_index
                )
                package.global_shaping[env_index] = (
                    torch.linalg.vector_norm(
                        package.state.pos[env_index] - package.goal.state.pos[env_index]
                    )
                    * self.shaping_factor
                )


    def reward(self, agent: Agent):
        is_first = agent == self.world.agents[0]

        if is_first:
            self.rew = torch.zeros(
                self.world.batch_dim,
                device=self.world.device,
                dtype=torch.float32,
            )

            for package in self.packages:
                package.dist_to_goal = torch.linalg.vector_norm(
                    package.state.pos - package.goal.state.pos, dim=1
                )
                package.on_goal = self.world.is_overlapping(package, package.goal)
                # Update package color based on whether it's on the goal
                current_color_val = (
                    Color.GREEN.value if package.on_goal.any() else Color.RED.value # Simplified for batch
                )
                package.color = torch.tensor(
                    current_color_val, # Use determined color
                    device=self.world.device,
                    dtype=torch.float32,
                ).repeat(self.world.batch_dim, 1)
                # More precise batch-wise color update:
                package.color[package.on_goal] = torch.tensor(Color.GREEN.value, device=self.world.device, dtype=torch.float32)
                package.color[~package.on_goal] = torch.tensor(Color.RED.value, device=self.world.device, dtype=torch.float32)


                package_shaping = package.dist_to_goal * self.shaping_factor
                # Positive reward for reducing distance, only if not already on goal
                self.rew[~package.on_goal] += (
                    package.global_shaping[~package.on_goal]
                    - package_shaping[~package.on_goal]
                )
                package.global_shaping = package_shaping

        return self.rew

    def observation(self, agent: Agent):
        # get positions of all entities in this agent's reference frame
        package_obs_parts = []
        for package in self.packages:
            package_obs_parts.append(package.state.pos - package.goal.state.pos) # Relative package to goal
            package_obs_parts.append(package.state.pos - agent.state.pos)      # Relative package to agent
            package_obs_parts.append(package.state.vel)
            package_obs_parts.append(package.on_goal.unsqueeze(-1).float()) # .float() for tensor concat

        # Lidar observations
        # agent.sensors[0] is agent_lidar
        # agent.sensors[1] is package_lidar
        # agent.sensors[2] is obstacle_lidar
        lidar_agents_measures = agent.sensors[0].measure()
        lidar_packages_measures = agent.sensors[1].measure()
        lidar_obstacles_measures = agent.sensors[2].measure()

        return torch.cat(
            [
                agent.state.pos,
                agent.state.vel,
                *package_obs_parts,
                lidar_agents_measures,
                lidar_packages_measures,
                lidar_obstacles_measures,
            ],
            dim=-1,
        )

    def done(self):
        # Done if all packages are on their respective goals
        return torch.all(
            torch.stack(
                [package.on_goal for package in self.packages],
                dim=1, # stack along a new dimension
            ),
            dim=-1, # check .all() along that new dimension
        )


class HeuristicPolicy(BaseHeuristicPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lookahead = 0.0
        self.start_vel_dist_from_target_ratio = 0.5
        self.start_vel_behind_ratio = 0.5
        self.start_vel_mag = 1.0
        self.hit_vel_mag = 1.0
        self.package_radius = 0.15 / 2 # Matching package.shape.radius for Box (approx)
        self.agent_radius = 0.03 # Matching agent.shape.radius
        self.dribble_slowdown_dist = 0.0
        self.speed = 0.95

    def compute_action(self, observation: torch.Tensor, u_range: float) -> torch.Tensor:
        # The observation structure has changed due to LIDARs.
        # The heuristic needs to be updated if it relies on specific indices
        # For now, let's make a simple heuristic: move towards the first package.
        # obs layout:
        # agent.state.pos (2)
        # agent.state.vel (2)
        # For each package:
        #   package.state.pos - package.goal.state.pos (2)
        #   package.state.pos - agent.state.pos (2)
        #   package.state.vel (2)
        #   package.on_goal.unsqueeze(-1) (1)
        # lidar_agents_measures (n_lidar_rays_agents)
        # lidar_packages_measures (n_lidar_rays_packages)
        # lidar_obstacles_measures (n_lidar_rays_obstacles)

        self.n_env = observation.shape[0]
        self.device = observation.device
        agent_pos = observation[:, :2]

        # Assuming 1 package for simplicity in this heuristic update
        if len(self.scenario.packages) > 0: # Use self.scenario to access scenario attributes
            # Index for the first package's relative position to agent
            # agent_pos (2) + agent_vel (2) + package_to_goal (2) = 6
            # So package_to_agent starts at index 6
            relative_package_pos = observation[:, 6:8] # This is (package_pos - agent_pos)
            target_pos_for_agent = agent_pos + relative_package_pos # This is package_pos

            # A very simple action: move towards the package
            direction_to_package = relative_package_pos
            control = direction_to_package * self.speed # Scale by speed
        else:
            # No packages, maybe move randomly or stay still
            control = torch.randn_like(agent_pos) * 0.1

        control *= u_range # Apply u_range scaling generally
        return torch.clamp(control, -u_range, u_range)

    # The dribble, hermite, nPr, get_start_vel, get_action methods from the original
    # heuristic are quite complex and specific to a "dribbling" behavior towards a single target.
    # Adapting them robustly to multiple packages, LIDAR data, and obstacles would require
    # significant changes. The simplified heuristic above is just a placeholder.
    # If you need the advanced dribbling, it would need to be refactored to select a target package
    # and use LIDAR to avoid obstacles/other agents while approaching.


if __name__ == "__main__":
    # Pass the updated scenario class to render_interactively
    render_interactively(
        Scenario(), # Pass an instance of the scenario
        # Add any kwargs your scenario's make_world now expects, e.g.:
        n_agents=2,
        n_packages=1,
        n_obstacles=3,
        lidar_range=0.5,
        # control_two_agents=True, # if n_agents >= 2
    )