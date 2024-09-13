# main big2PPOSimulation class

import numpy as np
from rl_models.PPONetwork_torch import PPONetwork, PPOModel, NeuRDNetwork, NeuRDSequentialNetwork
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


def flatten01(arr):
    """
    flatten axes 0 and 1
    """
    s = arr.shape
    return arr.reshape(s[0] * s[1], *s[2:])


class big2PPOSimulation(object):
    def __init__(
        self,
        *,
        inpDim=412,
        nGames=8,
        nSteps=20,
        nMiniBatches=4,
        nOptEpochs=1,
        lam=0.95,
        ent_coef=0.001,
        vf_coef=0.5,
        max_grad_norm=0.5,
        minLearningRate=0.000001,
        learningRate,
        clipRange,
        vfClipRange,
        saveEvery=1000,
        l2_coef=1e-4,
        variants="neurd",
    ):
        # network/model for training
        # self.trainingNetwork = NeuRDNetwork(
        #     inpDim, 1695, {"pi": [512, 512], "vf": [256]}, nn.ReLU, "cuda"
        # )

        self.trainingNetwork = NeuRDSequentialNetwork(
            inpDim + 22 * 13 + 1695,
            1695,
            {"pi": [512, 512], "vf": [256]},
            nn.ReLU,
            "cuda",
        )
        self.trainingModel = PPOModel(
            self.trainingNetwork,
            inpDim,
            1695,
            ent_coef,
            vf_coef,
            max_grad_norm,
            l2_coef,
        )

        # player networks which choose decisions - allowing for later on experimenting with playing against older versions of the network (so decisions they make are not trained on).
        self.playerNetworks = {}

        # for now each player uses the same (up to date) network to make it's decisions.
        self.playerNetworks[1] = self.playerNetworks[2] = self.playerNetworks[
            3
        ] = self.playerNetworks[4] = self.trainingNetwork
        self.trainOnPlayer = [True, True, True, True]

        # environment
        self.vectorizedGame = vectorizedBig2Games(nGames)

        # params
        self.nGames = nGames
        self.inpDim = inpDim
        self.nSteps = nSteps
        self.nMiniBatches = nMiniBatches
        self.nOptEpochs = nOptEpochs
        self.lam = lam
        self.learningRate = learningRate
        self.minLearningRate = minLearningRate
        self.clipRange = clipRange
        self.vfClipRange = vfClipRange
        self.saveEvery = saveEvery
        self.variants = variants

        self.rewardNormalization = (
            30.0  # divide rewards by this number (so reward ranges from -1.0 to 3.0)
        )

        # test networks - keep network saved periodically and run test games against current network
        self.testNetworks = {}

        # final 4 observations need to be carried over (for value estimation and propagating rewards back)
        self.prevObs = []
        self.prevGos = []
        self.prevAvailAcs = []
        self.prevActionFeats = []
        self.prevRewards = []
        self.prevActions = []
        self.prevValues = []
        self.prevDones = []
        self.prevNeglogpacs = []

        # episode/training information
        self.totTrainingSteps = 0
        self.epInfos = []
        self.gamesDone = 0
        self.losses = []

    @th.no_grad()
    def run(self):
        # run vectorized games for nSteps and generate mini batch to train on.
        (
            mb_obs,
            mb_pGos,
            mb_actions,
            mb_values,
            mb_neglogpacs,
            mb_rewards,
            mb_dones,
            mb_availAcs,
            mb_availAcsFeats,  # n_steps, n_games, n_avail_actions, action_feat_dim
        ) = ([], [], [], [], [], [], [], [], [])
        for i in range(len(self.prevObs)):
            mb_obs.append(self.prevObs[i])
            mb_pGos.append(self.prevGos[i])
            mb_actions.append(self.prevActions[i])
            mb_values.append(self.prevValues[i])
            mb_neglogpacs.append(self.prevNeglogpacs[i])
            mb_rewards.append(self.prevRewards[i])
            mb_dones.append(self.prevDones[i])
            mb_availAcs.append(self.prevAvailAcs[i])
            mb_availAcsFeats.append(self.prevActionFeats[i])
        if len(self.prevObs) == 4:
            endLength = self.nSteps
        else:
            endLength = self.nSteps - 4
        for _ in range(self.nSteps):
            (
                currGos,
                currStates,
                currAvailAcs,
                currAvailAcsFeats,
            ) = self.vectorizedGame.getCurrStates()
            currStates = np.squeeze(currStates)
            currAvailAcs = np.squeeze(currAvailAcs)
            currGos = np.squeeze(currGos)
            actions, values, neglogpacs = self.trainingNetwork.step(
                th.from_numpy(currStates).float().cuda(),
                th.from_numpy(currAvailAcs).float().cuda(),
                [
                    th.from_numpy(currAvailAcsFeat).float().cuda()
                    for currAvailAcsFeat in currAvailAcsFeats
                ],
            )
            actions = actions.cpu().numpy()
            values = values.cpu().numpy().flatten()
            neglogpacs = neglogpacs.cpu().numpy()
            rewards, dones, infos = self.vectorizedGame.step(actions)
            mb_obs.append(currStates.copy())
            mb_pGos.append(currGos)
            mb_availAcs.append(currAvailAcs.copy())
            mb_availAcsFeats.append(currAvailAcsFeats.copy())
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(list(dones))
            # now back assign rewards if state is terminal
            toAppendRewards = np.zeros((self.nGames,))
            mb_rewards.append(toAppendRewards)
            for i in range(self.nGames):
                if dones[i] == True:
                    reward = rewards[i]
                    mb_rewards[-1][i] = (
                        reward[mb_pGos[-1][i] - 1] / self.rewardNormalization
                    )
                    mb_rewards[-2][i] = (
                        reward[mb_pGos[-2][i] - 1] / self.rewardNormalization
                    )
                    mb_rewards[-3][i] = (
                        reward[mb_pGos[-3][i] - 1] / self.rewardNormalization
                    )
                    mb_rewards[-4][i] = (
                        reward[mb_pGos[-4][i] - 1] / self.rewardNormalization
                    )
                    mb_dones[-2][i] = True
                    mb_dones[-3][i] = True
                    mb_dones[-4][i] = True
                    # self.epInfos.append(infos[i])
                    self.gamesDone += 1
                    # print("Game %d finished. Lasted %d turns" % (self.gamesDone, infos[i]['numTurns']))
        self.prevObs = mb_obs[endLength:]
        self.prevGos = mb_pGos[endLength:]
        self.prevRewards = mb_rewards[endLength:]
        self.prevActions = mb_actions[endLength:]
        self.prevValues = mb_values[endLength:]
        self.prevDones = mb_dones[endLength:]
        self.prevNeglogpacs = mb_neglogpacs[endLength:]
        self.prevAvailAcs = mb_availAcs[endLength:]
        self.prevActionFeats = mb_availAcsFeats[endLength:]
        mb_obs = np.asarray(mb_obs, dtype=np.float32)[:endLength]
        mb_availAcs = np.asarray(mb_availAcs, dtype=np.float32)[:endLength]
        mb_availAcsFeats = np.asarray(mb_availAcsFeats, dtype=object)[:endLength]
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)[:endLength]
        mb_actions = np.asarray(mb_actions, dtype=np.float32)[:endLength]
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)[:endLength]
        mb_dones = np.asarray(mb_dones, dtype=bool)
        # discount/bootstrap value function with generalized advantage estimation:
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        for k in range(4):
            lastgaelam = 0
            for t in reversed(range(k, endLength, 4)):
                nextNonTerminal = 1.0 - mb_dones[t]
                nextValues = mb_values[t + 4]
                delta = mb_rewards[t] + nextValues * nextNonTerminal - mb_values[t]
                mb_advs[t] = lastgaelam = (
                    delta + self.lam * nextNonTerminal * lastgaelam
                )

        mb_values = mb_values[:endLength]
        # mb_dones = mb_dones[:endLength]
        mb_returns = mb_advs + mb_values

        # return (mb_obs, mb_availAcs, mb_availAcsFeats, mb_returns, mb_actions, mb_values, mb_neglogpacs)
        return map(
            sf01,
            (
                mb_obs,
                mb_availAcs,
                mb_availAcsFeats,
                mb_returns,
                mb_actions,
                mb_values,
                mb_neglogpacs,
            ),
        )

    def train(self, nTotalSteps):
        nUpdates = nTotalSteps // (self.nGames * self.nSteps)

        pbar = tqdm(range(nUpdates))
        for update in pbar:
            alpha = 1.0 - update / nUpdates
            lrnow = self.learningRate * alpha
            if lrnow < self.minLearningRate:
                lrnow = self.minLearningRate
            pbar.set_description("Games Done: %d" % self.gamesDone, "LR: %f" % lrnow)

            cliprangenow = self.clipRange * alpha
            vfcliprangenow = self.vfClipRange * alpha
            (
                states,
                availAcs,
                actionFeats,
                returns,
                actions,
                values,
                neglogpacs,
            ) = self.run()

            batchSize = states.shape[0]
            self.totTrainingSteps += batchSize

            # nTrainingBatch = batchSize // self.nMiniBatches

            mb_lossvals = []
            if self.variants == "ppo":
                mb_lossvals.append(
                    self.trainingModel.train(
                        lrnow,
                        cliprangenow,
                        vfcliprangenow,
                        states,
                        availAcs,
                        actionFeats,
                        returns,
                        actions,
                        values,
                        neglogpacs,
                    )
                )
            elif self.variants == "neurd":
                mb_lossvals.append(
                    self.trainingModel.train_neurd(
                        lrnow,
                        cliprangenow,
                        vfcliprangenow,
                        states,
                        availAcs,
                        actionFeats,
                        returns,
                        actions,
                        values,
                        neglogpacs,
                    )
                )
            # inds = np.arange(batchSize)
            # for _ in range(self.nOptEpochs):
            #     np.random.shuffle(inds)
            #     for start in range(0, batchSize, nTrainingBatch):
            #         end = start + nTrainingBatch
            #         mb_inds = inds[start:end]
            #         if self.variants == "ppo":
            #             mb_lossvals.append(
            #                 self.trainingModel.train(
            #                     lrnow,
            #                     cliprangenow,
            #                     vfcliprangenow,
            #                     states[mb_inds],
            #                     availAcs[mb_inds],
            #                     actionFeats[mb_inds],
            #                     returns[mb_inds],
            #                     actions[mb_inds],
            #                     values[mb_inds],
            #                     neglogpacs[mb_inds],
            #                 )
            #             )
            #         elif self.variants == "neurd":
            #             mb_lossvals.append(
            #                 self.trainingModel.train_neurd(
            #                     lrnow,
            #                     cliprangenow,
            #                     vfcliprangenow,
            #                     states[mb_inds],
            #                     availAcs[mb_inds],
            #                     actionFeats[mb_inds],
            #                     returns[mb_inds],
            #                     actions[mb_inds],
            #                     values[mb_inds],
            #                     neglogpacs[mb_inds],
            #                 )
            #             )

            lossvals = np.mean(mb_lossvals, axis=0)
            self.losses.append(lossvals)

            # Check nan parameters
            is_nan = th.stack(
                [th.any(th.isnan(p)) for p in self.trainingNetwork.parameters()]
            ).any()
            if is_nan:
                print("Nan parameters detected. Stopping training.")
                break

            if update % self.saveEvery == 0:
                # print parameters average absolute value and max absolute value
                for name, p in self.trainingNetwork.named_parameters():
                    print(
                        "Name: %s, Mean: %f, Max: %f"
                        % (name, th.mean(th.abs(p)), th.max(th.abs(p)))
                    )

                name = "modelParameters_th_" + str(update)
                th.save(self.trainingNetwork.state_dict(), name)
                joblib.dump(self.losses, "losses_th.pkl")
                # joblib.dump(self.epInfos, "epInfos.pkl")


if __name__ == "__main__":
    import time

    th.autograd.set_detect_anomaly(False)
    mainSim = big2PPOSimulation(
        nGames=64,
        nSteps=20,
        learningRate=0.00005,
        clipRange=0.2,
        vfClipRange=0.1,
        vf_coef=0.5,
        ent_coef=0.001,
        l2_coef=1e-5,
        variants="neurd",
    )
    start = time.time()
    mainSim.train(1000000000)
    end = time.time()
    print("Time Taken: %f" % (end - start))
