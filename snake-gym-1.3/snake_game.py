import gym
from gym import spaces
import pygame
import numpy as np
from operator import add


class SnakeGameEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"], 
        "output": ["features", "image"],
        "render_fps": 1}

    def __init__(self, render_mode=None, output="features", size=20):
        self.size = size  # The size of the square grid
        self.window_size = 420  # The size of the PyGame window
        self.output = output

        # Load assets
        self.background_img = pygame.image.load('img/background.png')
        self.snake_img = pygame.image.load('img/snakeBody.png')
        self.food_img = pygame.image.load('img/food2.png')
        
        if output=="image":
            self.observation_space = spaces.Box(low=0, high=255, shape=[self.window_size, self.window_size, 3], dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(low=0, high=1, shape=(11,), dtype=np.int8)
        # We have 3 actions, corresponding to
        self.action_space = spaces.Discrete(3)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        # self._action_to_direction = {
        #     0: np.array([1, 0, 0]), #continue
        #     1: np.array([0, 1, 0]), #turn right
        #     2: np.array([0, 0, 1]), #turn left
        # }
        self._action_to_direction = {
            0: np.array([1, 0, 0]), #x, y right
            1: np.array([0, 1, 0]), #left
            2: np.array([0, 0, 1]), #turn up
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None


        # Init game logic
        self.score = 0
        self.food = 1
        self.crash = False
        self.eaten = False
        self.x_cur = 9
        self.y_cur = 9
        self.x_change = 1
        self.y_change = 0
        self.x_food = -1
        self.y_food = -1
        self.position = []
        self.position.append([self.x_cur, self.y_cur])
        self._spawn_food()


    def _get_obs(self):
        if self.output=='image':
            canvas = self.draw_canvas()
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        else:
            """
            Return the state.
            The state is a numpy array of 11 values, representing:
                - Danger 1 OR 2 steps ahead
                - Danger 1 OR 2 steps on the right
                - Danger 1 OR 2 steps on the left
                - Snake is moving left
                - Snake is moving right
                - Snake is moving up
                - Snake is moving down
                - The food is on the left
                - The food is on the right
                - The food is on the upper side
                - The food is on the lower side      
            """
            state = [
                (self.x_change == 1 and self.y_change == 0 and ((list(map(add, self.position[-1], [1, 0])) in self.position) or
                self.position[-1][0] + 1 >= 19)) 
                or (self.x_change == -1 and self.y_change == 0 and ((list(map(add, self.position[-1], [-1, 0])) in self.position) or
                self.position[-1][0] - 1 < 1)) 
                or (self.x_change == 0 and self.y_change == -1 and ((list(map(add, self.position[-1], [0, -1])) in self.position) or
                self.position[-1][-1] - 1 < 1)) 
                or (self.x_change == 0 and self.y_change == 1 and ((list(map(add, self.position[-1], [0, 1])) in self.position) or
                self.position[-1][-1] + 1 >= 19)),  # danger straight

                (self.x_change == 0 and self.y_change == -1 and ((list(map(add,self.position[-1],[1, 0])) in self.position) or
                self.position[ -1][0] + 1 >= 19)) 
                or (self.x_change == 0 and self.y_change == 1 and ((list(map(add,self.position[-1], [-1,0])) in self.position) or self.position[-1][0] - 1 <= 0)) 
                or (self.x_change == -1 and self.y_change == 0 and ((list(map(
                add,self.position[-1],[0,-1])) in self.position) or 
                self.position[-1][-1] - 1 < 1))
                or (self.x_change == 1 and self.y_change == 0 and ((list(map(add,self.position[-1],[0,1])) in self.position) or 
                self.position[-1][-1] + 1 >= 19)),  # danger right

                (self.x_change == 0 and self.y_change == 1 and ((list(map(add,self.position[-1],[1,0])) in self.position) or
                self.position[-1][0] + 1 >= 19)) 
                or (self.x_change == 0 and self.y_change == -1 and ((list(map(
                add, self.position[-1],[-1,0])) in self.position) or self.position[-1][0] - 1 < 1)) 
                or (self.x_change == 1 and self.y_change == 0 and (
                (list(map(add,self.position[-1],[0,-1])) in self.position) or self.position[-1][-1] - 1 < 1)) 
                or (self.x_change == -1 and self.y_change == 0 and ((list(map(add,self.position[-1],[0,1])) in self.position) or
                self.position[-1][-1] + 1 >= 19)), #danger left

                
                self.x_change == -1,  # move left
                self.x_change == 1,  # move right
                self.y_change == -1,  # move up
                self.y_change == 1,  # move down
                self.x_food < self.x_cur,  # food left
                self.x_food > self.x_cur,  # food right
                self.y_food < self.y_cur,  # food up
                self.y_food > self.y_cur  # food down
            ]
            return np.asarray(state, dtype=np.float32)

    def _get_info(self):
        return {
            "score": self.score
        }

    def _spawn_food(self):
        x_rand = np.random.randint(20)
        y_rand = np.random.randint(20)

        if [x_rand, y_rand] in self.position:
            self._spawn_food()
        else:
            self.x_food = x_rand
            self.y_food = y_rand
            return x_rand, y_rand

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Choose the agent's location 
        self.x_cur = 9
        self.y_cur = 9
        self.x_change = 1
        self.y_change = 0
        self.food = 1
        self.position = []
        self.position.append([self.x_cur, self.y_cur])

        # Randomly spawn food
        self._spawn_food()
        # Set terminate False not sure if needed
        self.crash = False
        self.eaten = False

        # Set score back to 00
        self.score = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2}) to the direction we walk in
        direction = self._action_to_direction[action]

        # Game logic

        if self.eaten:
            self.position.append([self.x_cur, self.y_cur])
            self.eaten = False
            self.food = self.food + 1

        # Egocentric to cartesian coord
        move_array = [self.x_change, self.y_change]
        if np.array_equal(direction, [1, 0, 0]):
            move_array = self.x_change, self.y_change
        elif np.array_equal(direction, [0, 1, 0]) and self.y_change == 0:  
            # right - going horizontal
            move_array = [0, self.x_change]
        elif np.array_equal(direction, [0, 1, 0]) and self.x_change == 0:  
            # right - going vertical
            move_array = [-self.y_change, 0]
        elif np.array_equal(direction, [0, 0, 1]) and self.y_change == 0:  
            # left - going horizontal
            move_array = [0, -self.x_change]
        elif np.array_equal(direction, [0, 0, 1]) and self.x_change == 0:  
            # left - going vertical
            move_array = [self.y_change, 0]

        self.x_change, self.y_change = move_array
        self.x_cur = self.x_cur + self.x_change
        self.y_cur = self.y_cur + self.y_change

        #self.x_cur = self.x_cur + direction[0]
        #self.y_cur = self.y_cur + direction[1]
        # Set teriminate condition
        if self.x_cur < 0 or self.x_cur > 19 \
                or self.y_cur < 0 \
                or self.y_cur > 19 \
                or [self.x_cur, self.y_cur] in self.position:
            self.crash = True

        # Set eat condition
        if self.x_cur == self.x_food and self.y_cur == self.y_food:
            self._spawn_food()
            self.eaten = True
            self.score = self.score + 1

        if self.position[-1][0] != self.x_cur or self.position[-1][1] != self.y_cur:
            if self.food > 1:
                for i in range(0, self.food - 1):
                    self.position[i][0], self.position[i][1] = self.position[i + 1]
            self.position[-1][0] = self.x_cur
            self.position[-1][1] = self.y_cur

        # An episode is done iff the agent has reached the target
        terminated = self.crash
        reward = 0
        if self.crash:
            reward = -10
        if self.eaten:
            reward = 10

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def draw_canvas(self):
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        pix_square_size = 20
        canvas.blit(canvas, canvas.get_rect())
        canvas.blit(self.background_img, (0, 0))

        # draw each snake
        if self.crash is False:
            for i in range(self.food):
                x_temp, y_temp = self.position[len(self.position) - 1 - i]
                canvas.blit(self.snake_img, (10+x_temp*20, 10+y_temp*20))
    
        # draw food
        canvas.blit(self.food_img, (10+self.x_food*20, 10+self.y_food*20))
        return canvas

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = self.draw_canvas()


        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())

            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
