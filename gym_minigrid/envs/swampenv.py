import numpy as np

from gym_minigrid.minigrid import Goal, Grid, Swamp, MiniGridEnv, MissionSpace, Lava, Lava2


class SwampEnv(MiniGridEnv):
    """
    This is the environment used for Causal MoMa
    It contains three different types of tiles, where arm actions are needed to avoid penalty
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

    def get_step_reward(self, cur_cell, fwd_cell, arm_l, arm_r, locomotion_dir, non_empty_count, done):
        ################# reward initialization block  ####################
        swamp_reward = 0
        locomotion_reward = locomotion_dir
        arm_l_collision_reward = 0
        arm_r_collision_reward = 0

        ##############  arm collision block  ###################
        if cur_cell is not None and cur_cell.type == "lava":
            if arm_l != non_empty_count % 3:
                arm_l_collision_reward = -5
        if cur_cell is not None and cur_cell.type == "lava2":
            if arm_r != non_empty_count % 3:
                arm_r_collision_reward = -5
        if fwd_cell is not None and fwd_cell.type == "swamp":
            swamp_reward = -5

        ##############  reward total block  ###################
        total_reward = np.concatenate(
            [locomotion_reward, [swamp_reward, arm_l_collision_reward, arm_r_collision_reward]])

        return total_reward

