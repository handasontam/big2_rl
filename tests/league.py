# main big2PPOSimulation class

import numpy as np
from rl_models.PPONetwork_torch import PPONetwork, NeuRDNetwork, NeuRDSequentialNetwork
from big2Game_torch import vectorizedBig2Games
import joblib
import copy
import os
import time
import torch as th
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm


# taken directly from baselines implementation - reshape minibatch in preparation for training.
def sf01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    s = arr.shape
    return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])


class big2PPOSimulation(object):
    def __init__(self, *, playerNetworks=[], inpDim=412, nSteps=20, saveEvery=1000):
        # #network/model for training
        # self.trainingNetwork1 = PPONetwork(inpDim, 1695, {"pi": [512,512], "vf": [256]}, nn.ReLU, "cuda")
        # self.trainingNetwork2 = PPONetwork(inpDim, 1695, {"pi": [512,512], "vf": [256]}, nn.ReLU, "cuda")
        # self.trainingNetwork3 = PPONetwork(inpDim, 1695, {"pi": [512,512], "vf": [256]}, nn.ReLU, "cuda")
        # self.trainingNetwork4 = PPONetwork(inpDim, 1695, {"pi": [512,512], "vf": [256]}, nn.ReLU, "cuda")

        # self.trainingNetwork1.load_state_dict(th.load(model_parameters_path[0], map_location='cuda'))
        # self.trainingNetwork2.load_state_dict(th.load(model_parameters_path[1], map_location='cuda'))
        # self.trainingNetwork3.load_state_dict(th.load(model_parameters_path[2], map_location='cuda'))
        # self.trainingNetwork4.load_state_dict(th.load(model_parameters_path[3], map_location='cuda'))

        # self.trainingNetwork1.eval()
        # self.trainingNetwork2.eval()
        # self.trainingNetwork3.eval()
        # self.trainingNetwork4.eval()

        # #player networks which choose decisions - allowing for later on experimenting with playing against older versions of the network (so decisions they make are not trained on).
        # self.playerNetworks = {}

        # #for now each player uses the same (up to date) network to make it's decisions.
        # self.playerNetworks[1] = self.trainingNetwork1
        # self.playerNetworks[2] = self.trainingNetwork2
        # self.playerNetworks[3] = self.trainingNetwork3
        # self.playerNetworks[4] = self.trainingNetwork4
        self.playerNetworks = playerNetworks

        # environment
        self.vectorizedGame = vectorizedBig2Games(1)

        # params
        self.inpDim = inpDim
        self.nSteps = nSteps
        self.saveEvery = saveEvery

        # episode/training information
        self.totTrainingSteps = 0
        self.epInfos = []
        self.gamesDone = 0
        self.losses = []

    @th.no_grad()
    def run(self):
        # run vectorized games for nSteps and generate mini batch to train on.
        for _ in range(self.nSteps):
            (
                currGos,
                currStates,
                currAvailAcs,
                currAvailAcsFeats,
            ) = self.vectorizedGame.getCurrStates()
            currStates = np.squeeze(currStates)
            currAvailAcs = np.squeeze(currAvailAcs)
            currGo = currGos[0]
            actions, _, _ = self.playerNetworks[currGo].step(
                th.from_numpy(currStates).unsqueeze(0).float().cuda(),
                th.from_numpy(currAvailAcs).unsqueeze(0).float().cuda(),
                [th.from_numpy(currAvailAcsFeat).float().cuda() for currAvailAcsFeat in currAvailAcsFeats],
            )
            actions = actions.cpu().numpy()
            rewards, dones, infos = self.vectorizedGame.step(actions)
            for i in range(1):
                if dones[i] == True:
                    self.epInfos.append(infos[i])
                    self.gamesDone += 1
                    print(
                        "Game %d finished. Lasted %d turns"
                        % (self.gamesDone, infos[i]["numTurns"])
                    )
        return

    def train(self):
        pbar = tqdm(range(100000000))
        for update in pbar:
            pbar.set_description("Games Done: %d" % self.gamesDone)
            self.run()
            if self.gamesDone >= 3000:
                break

        # print(self.epInfos)
        print(
            "Average Num Turns: %f"
            % (np.mean([epInfo["numTurns"] for epInfo in self.epInfos]))
        )
        print(
            "Average reward for Player 1: ",
            np.mean([epInfo["rewards"][0] for epInfo in self.epInfos]),
        )
        print(
            "Average reward for Player 2: ",
            np.mean([epInfo["rewards"][1] for epInfo in self.epInfos]),
        )
        print(
            "Average reward for Player 3: ",
            np.mean([epInfo["rewards"][2] for epInfo in self.epInfos]),
        )
        print(
            "Average reward for Player 4: ",
            np.mean([epInfo["rewards"][3] for epInfo in self.epInfos]),
        )
        print(
            "Standard deviation for Player 1",
            np.std([epInfo["rewards"][0] for epInfo in self.epInfos]),
        )
        print(
            "Standard deviation for Player 2",
            np.std([epInfo["rewards"][1] for epInfo in self.epInfos]),
        )
        print(
            "Standard deviation for Player 3",
            np.std([epInfo["rewards"][2] for epInfo in self.epInfos]),
        )
        print(
            "Standard deviation for Player 4",
            np.std([epInfo["rewards"][3] for epInfo in self.epInfos]),
        )
        # joblib.dump(self.epInfos, "epInfos.pkl")


if __name__ == "__main__":
    import time

    th.autograd.set_detect_anomaly(True)
    # mainSim = big2PPOSimulation(nGames=64, nSteps=40, learningRate = 0.00025, clipRange = 0.2, vfClipRange = 0.1, l2_coef=1e-5)
    # model_parameters = ['vanilla_10000', 'modelParameters_th_119000', 'vanilla_10000', 'modelParameters_th_119000']
    # model_parameters = ['league/' + p for p in model_parameters]

    # network/model for training
    trainingNetwork1 = PPONetwork(
        412, 1695, {"pi": [512, 512], "vf": [256]}, nn.ReLU, "cuda"
    )
    # trainingNetwork2 = NeuRDNetwork(
    #     412, 1695, {"pi": [512, 512], "vf": [256]}, nn.ReLU, "cuda"
    # )
    trainingNetwork2 = NeuRDSequentialNetwork(
        412+22*13+1695, 1695, {"pi": [512, 512], "vf": [256]}, nn.ReLU, "cuda"
    )
    trainingNetwork3 = PPONetwork(
        412, 1695, {"pi": [512, 512], "vf": [256]}, nn.ReLU, "cuda"
    )
    trainingNetwork4 = NeuRDSequentialNetwork(
        412+22*13+1695, 1695, {"pi": [512, 512], "vf": [256]}, nn.ReLU, "cuda"
    )
    print(f"==>> trainingNetwork1: {trainingNetwork1}")
    print(f"==>> trainingNetwork2: {trainingNetwork2}")
    print(f"==>> trainingNetwork3: {trainingNetwork3}")
    print(f"==>> trainingNetwork4: {trainingNetwork4}")

    trainingNetwork1.load_state_dict(
        th.load("league/vanilla/vanilla_10000", map_location="cuda")
    )
    trainingNetwork2.load_state_dict(
        th.load("league/NeuRD_RichActsFeats/modelParameters_th_268000", map_location="cuda")
    )
    trainingNetwork3.load_state_dict(
        th.load("league/vanilla/vanilla_10000", map_location="cuda")
    )
    trainingNetwork4.load_state_dict(
        th.load("league/NeuRD_RichActsFeats/modelParameters_th_60000", map_location="cuda")
    )

    trainingNetwork1.eval()
    trainingNetwork2.eval()
    trainingNetwork3.eval()
    trainingNetwork4.eval()

    # player networks which choose decisions - allowing for later on experimenting with playing against older versions of the network (so decisions they make are not trained on).
    playerNetworks = {}

    # for now each player uses the same (up to date) network to make it's decisions.
    playerNetworks[1] = trainingNetwork1
    playerNetworks[2] = trainingNetwork2
    playerNetworks[3] = trainingNetwork3
    playerNetworks[4] = trainingNetwork4

    mainSim = big2PPOSimulation(
        playerNetworks=playerNetworks, inpDim=412, nSteps=3000, saveEvery=1000
    )
    start = time.time()
    mainSim.train()
    end = time.time()
    print("Time Taken: %f" % (end - start))
