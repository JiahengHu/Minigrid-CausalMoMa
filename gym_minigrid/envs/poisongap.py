# This is very similar to lavagap, except that the environment doesn't end when the robot enters the lava

# HERE: TODO: make this the gym environment

import numpy as np

from gym_minigrid.minigrid import Goal, Grid, Swamp, MiniGridEnv, MissionSpace, Lava, Lava2


class SwampEnv(MiniGridEnv):
    """
    Environment with one wall of lava with a small gap to cross through
    This environment is similar to LavaCrossing but simpler in structure.
    """

    def __init__(self, size, obstacle_type=Swamp, **kwargs):
        self.obstacle_type = obstacle_type
        self.size = size

        if obstacle_type == Swamp:
            mission_space = MissionSpace(
                mission_func=lambda: "avoid the Swamp and get to the green goal square"
            )
        else:
            mission_space = MissionSpace(
                mission_func=lambda: "find the opening and get to the green goal square"
            )

        super().__init__(
            mission_space=mission_space,
            width=size,
            height=size,
            max_steps=10,
            # Set this to True for maximum speed
            see_through_walls=True,
        )

    def _gen_grid(self, width, height):
        assert width >= 5 and height >= 5

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the top-left corner
        self.agent_pos = np.array((1, 1))
        self.agent_dir = 0

        # Place a goal square in the bottom-right corner
        self.goal_pos = np.array((width - 2, height - 2))
        self.put_obj(Goal(), *self.goal_pos)

        # Generate and store random gap position
        self.gap_pos = np.array(
            (
                self._rand_int(2, width - 2),
                self._rand_int(1, height - 1),
            )
        )

        #TODO: randomize location of lava

        # # Place the obstacle wall
        # self.grid.vert_wall(self.gap_pos[0], 1, height - 2, self.obstacle_type)
        #
        # # Put a hole in the wall
        # self.grid.set(*self.gap_pos, None)

        # Place the swamp
        self.n_obstacles = 5
        for i_obst in range(self.n_obstacles):
            # self.obstacles.append(self.obstacle_type)
            self.place_obj(self.obstacle_type(), max_tries=100)

        arm_obs = 8
        # Place the l_arm obs
        for i_obst in range(arm_obs):
            # self.obstacles.append(self.obstacle_type)
            self.place_obj(Lava(), max_tries=100)

        arm_obs = 8
        # Place the l_arm obs
        for i_obst in range(arm_obs):
            # self.obstacles.append(self.obstacle_type)
            self.place_obj(Lava2(), max_tries=100)

        self.mission = (
            "avoid the swamp and get to the green goal square"
            if self.obstacle_type == Swamp
            else "find the opening and get to the green goal square"
        )
