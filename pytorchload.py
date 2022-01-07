import retro
import neat
import pickle
import numpy as np

def replay_genome(config_path, genome_path="winner.pkl"):
    # Load requried NEAT config
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # Unpickle saved winner
    with open(genome_path, "rb") as f:
        genome = pickle.load(f)

    # Convert loaded genome into required data structure
    genomes = [(1, genome)]

    # Call game with only the loaded genome
    return genomes


env = retro.make('SonicTheHedgehog-Genesis', 'GreenHillZone.Act1')
ob = env.reset()

net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
imgarray = np.ndarray.flatten(ob)

nnOutput = net.activate(imgarray)

ob, rew, done, info = env.step(nnOutput)