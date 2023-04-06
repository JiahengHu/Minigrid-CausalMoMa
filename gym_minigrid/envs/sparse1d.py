import numpy as np

from gym_minigrid.minigrid import Goal, Grid, Swamp, MiniGridEnv, MissionSpace, Lava, Lava2


class Sparse1DEnv(MiniGridEnv):
    """
    This is the environment used for Causal MoMa - rebuttal
    1D env with delayed / undelayed reward
    Reward depends on arm_l at specific grid (4 steps before entering the grid)
    """

    def __init__(self, size, delayed, **kwargs):
        self.size = size
        self.delayed = delayed  # Whether the reward is delayed
        self.prev_arm_l = 0

        mission_space = MissionSpace(
            mission_func=lambda: "reach goal with specific arm action"
        )

        super().__init__(
            mission_space=mission_space,
            width=size,
            height=3,
            max_steps=10,
            # Set this to True for maximum speed
            see_through_walls=True,
        )

    def _gen_grid(self, width, height):
        assert width >= 3 and height >= 3

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

        self.mission = (
            "reach goal with specific arm action"
        )

    def get_step_reward(self, cur_cell, fwd_cell, arm_l, arm_r, locomotion_dir, non_empty_count, done):
        reach_reward = 0

        if done:
            if not self.delayed:
                important_arm_action = arm_l
            else:
                important_arm_action = self.prev_arm_l
            if important_arm_action == 0:
                reach_reward = 1

        total_reward = np.array([reach_reward])

        return total_reward

    def step(self, action):
        obs, reward, done, total_reward = super().step(action)
        self.prev_arm_l = action[2]
        return obs, reward, done, total_reward
