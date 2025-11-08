from src.gridworld import TextGridworld


def test_gridworld_greedy_reaches_goal():
    env = TextGridworld(size=3, max_steps=10, seed=0)
    obs = env.reset()
    assert "row" in obs
    steps = 0
    done = False
    while not done and steps < 10:
        action = env.greedy_action()
        obs, reward, done, info = env.step(action)
        steps += 1
    assert done or env.agent == env.goal
    assert info["distance"] >= 0
