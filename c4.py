from bindsnet.network import Network
from bindsnet.pipeline import EnvironmentPipeline
from bindsnet.learning import MSTDP
from bindsnet.encoding import bernoulli
from bindsnet.network.topology import Connection
from bindsnet.environment import GymEnvironment
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.pipeline.action import select_softmax
from gym_connect_four import ConnectFourEnv, Player, ResultType, SavedPlayer, RandomPlayer

# Build network.
network = Network(dt=1.0)

# Layers of neurons.
inpt = Input(n=7 * 6, shape=[1, 1, 1, 6, 7], traces=True)
middle = LIFNodes(n=420, traces=True) #TODO more layer?
out = LIFNodes(n=7, refrac=0, traces=True)

# Connections between layers.
inpt_middle = Connection(source=inpt, target=middle, wmin=0, wmax=1e-1)
middle_out = Connection(
    source=middle,
    target=out,
    wmin=0,
    wmax=1,
    update_rule=MSTDP,
    nu=1e-1,
    norm=0.5 * middle.n,
)

# Add all layers and connections to the network.
network.add_layer(inpt, name="Input Layer")
network.add_layer(middle, name="Hidden Layer")
network.add_layer(out, name="Output Layer")
network.add_connection(inpt_middle, source="Input Layer", target="Hidden Layer")
network.add_connection(middle_out, source="Hidden Layer", target="Output Layer")

# Load the Breakout environment.
environment = GymEnvironment("ConnectFour-v0")
environment.reset()

# Build pipeline from specified components.
environment_pipeline = EnvironmentPipeline(
    network,
    environment,
    encoding=bernoulli,
    action_function=select_softmax,
    output="Output Layer",
    time=100,
    history_length=1,
    delta=1,
    plot_interval=1,
    render_interval=1,
)


def run_pipeline(pipeline, episode_count):

    opponent = RandomPlayer(environment, 'OpponentRandomPlayer')
#    players = [player, opponent]
    one = 0
    two = 0
    for i in range(episode_count):
        total_reward = 0
        pipeline.reset_state_variables()
        is_done = False
        while not is_done:
            result = pipeline.env_step()
            pipeline.step(result)
            #TODO should i "perturb" state of pipeline with random player??
#            import pdb
#            pdb.set_trace()
#            pipeline.oppo_step(opponent)

            reward = result[1]
            total_reward += reward

            is_done = result[2]

        if result[1] == 1:
            one += 1
        else:
            two += 1
        print(f"Player {result[1]} Wins")
        print(f"Episode {i} total reward:{total_reward}")
    print("Player One Wins ", one)
    print("Player Two Wins ", two)

print("Training: ")
run_pipeline(environment_pipeline, episode_count=100)

# stop MSTDP
environment_pipeline.network.learning = False

print("Testing: ")
run_pipeline(environment_pipeline, episode_count=100)
