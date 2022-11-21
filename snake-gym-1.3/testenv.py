import gym
import snake_game


# env = gym.make('snake_gym/SnakeGame-v1', render_mode='human')
env = snake_game.SnakeGameEnv(render_mode='human', output="features")
env.reset()

while True:
    screen = env.render()
    action = 0
    inputs = input()
    if inputs == 'w':
        action = 0
    if inputs == 'a':
        action = 1
    if inputs == 'd':
        action = 2
    observation, reward, terminated, _, info = env.step(action)
    print(observation)

print(screen)