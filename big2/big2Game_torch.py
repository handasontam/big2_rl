# big 2 class
import big2.enumerateOptions as enumerateOptions
import big2.gameLogic as gameLogic
import numpy as np
import random
import math
from collections import defaultdict
from multiprocessing import Process, Pipe


class handPlayed:
    def __init__(self, hand, player):
        self.hand = hand
        self.player = player
        self.nCards = len(hand)
        if self.nCards <= 3:
            self.type = 1
        elif self.nCards == 4:
            if gameLogic.isFourOfAKind(hand):
                self.type = 2
            else:
                self.type = 1
        elif self.nCards == 5:
            if gameLogic.isStraight(hand):
                if gameLogic.isFlush(hand):
                    self.type = 4
                else:
                    self.type = 1
            elif gameLogic.isFlush(hand):
                self.type = 2
            else:
                self.type = 3


class big2Game:
    def __init__(self):
        self.reset()

    def reset(self):
        shuffledDeck = np.random.permutation(52) + 1
        # hand out cards to each player
        self.currentHands = {}
        self.currentHands[1] = np.sort(shuffledDeck[0:13])
        self.currentHands[2] = np.sort(shuffledDeck[13:26])
        self.currentHands[3] = np.sort(shuffledDeck[26:39])
        self.currentHands[4] = np.sort(shuffledDeck[39:52])
        self.cardsPlayed = np.zeros((4, 52), dtype=int)
        # who has 3D - this gets played
        for i in range(52):
            if shuffledDeck[i] == 1:
                threeDiamondInd = i
                break
        if threeDiamondInd < 13:
            whoHas3D = 1
        elif threeDiamondInd < 26:
            whoHas3D = 2
        elif threeDiamondInd < 39:
            whoHas3D = 3
        else:
            whoHas3D = 4
        self.whoHas3D = whoHas3D
        self.goIndex = 1
        self.handsPlayed = {}
        self.handsPlayed[self.goIndex] = handPlayed([], whoHas3D)
        self.goIndex += 1
        self.playersGo = whoHas3D
        if self.playersGo == 5:
            self.playersGo = 1
        self.passCount = 0
        self.control = 1
        self.neuralNetworkInputs = {}
        self.neuralNetworkInputs[1] = np.zeros(
            (412,), dtype=int
        )  # 412 dimensional state vector
        self.neuralNetworkInputs[2] = np.zeros((412,), dtype=int)
        self.neuralNetworkInputs[3] = np.zeros((412,), dtype=int)
        self.neuralNetworkInputs[4] = np.zeros((412,), dtype=int)

        nPlayerInd = 22 * 13  # see Figure 3
        nnPlayerInd = nPlayerInd + 27
        nnnPlayerInd = nnPlayerInd + 27
        # initialize number of cards left for each next player
        for i in range(1, 5):
            self.neuralNetworkInputs[i][nPlayerInd + 12] = 1
            self.neuralNetworkInputs[i][nnPlayerInd + 12] = 1
            self.neuralNetworkInputs[i][nnnPlayerInd + 12] = 1
        self.fillNeuralNetworkHand(1)
        self.fillNeuralNetworkHand(2)
        self.fillNeuralNetworkHand(3)
        self.fillNeuralNetworkHand(4)
        self.updateNeuralNetworkInputs(np.array([1]), whoHas3D)
        self.gameOver = 0
        self.rewards = np.zeros((4,))
        self.goCounter = 0

    def fillNeuralNetworkHand(self, player):
        handOptions = gameLogic.handsAvailable(self.currentHands[player])
        sInd = 0
        self.neuralNetworkInputs[player][sInd : 22 * 13] = 0
        # TODO: vectorize the below
        for cardID in self.currentHands[player]:
            value = handOptions.cards[cardID].value  # 1-13
            ###########################################################################
            self.neuralNetworkInputs[player][sInd + int(value) - 1] = 1
            if handOptions.cards[cardID].inStraight:
                self.neuralNetworkInputs[player][sInd + 20] = 1
            sInd += 22

        self.neuralNetworkInputs[player][
            np.arange(0, len(self.currentHands[player]) * 22, 22) + 13
        ] = (handOptions.suits==1).astype(int)
        self.neuralNetworkInputs[player][
            np.arange(0, len(self.currentHands[player]) * 22, 22) + 14
        ] = (handOptions.suits==2).astype(int)
        self.neuralNetworkInputs[player][
            np.arange(0, len(self.currentHands[player]) * 22, 22) + 15
        ] = (handOptions.suits==3).astype(int)
        self.neuralNetworkInputs[player][
            np.arange(0, len(self.currentHands[player]) * 22, 22) + 16
        ] = (handOptions.suits==0).astype(int)
    
        self.neuralNetworkInputs[player][
            np.arange(0, len(self.currentHands[player]) * 22, 22) + 17
        ] = handOptions.inPair
        self.neuralNetworkInputs[player][
            np.arange(0, len(self.currentHands[player]) * 22, 22) + 18
        ] = handOptions.inThreeOfAKind
        self.neuralNetworkInputs[player][
            np.arange(0, len(self.currentHands[player]) * 22, 22) + 19
        ] = handOptions.inFourOfAKind
        self.neuralNetworkInputs[player][
            np.arange(0, len(self.currentHands[player]) * 22, 22) + 21
        ] = handOptions.inFlush

    def updateNeuralNetworkPass(self, cPlayer):
        # current Player pass
        # this is a bit of a mess tbh, some things are unnecessary.
        phInd = (
            22 * 13 + 27 + 27 + 27 + 16
        )  # starting index of previous hand [phInd:phInd+29]
        nPlayer = cPlayer - 1
        if nPlayer == 0:
            nPlayer = 4
        nnPlayer = nPlayer - 1
        if nnPlayer == 0:
            nnPlayer = 4
        nnnPlayer = nnPlayer - 1
        if nnnPlayer == 0:
            nnnPlayer = 4
        if self.passCount < 2:
            # no control - prev hands remain same, i.e. keep[phInd:phInd+26]
            self.neuralNetworkInputs[nPlayer][phInd + 26 :] = 0
            self.neuralNetworkInputs[nnPlayer][phInd + 26 :] = 0
            self.neuralNetworkInputs[nnnPlayer][phInd + 26 :] = 0
            if self.passCount == 0:
                self.neuralNetworkInputs[nPlayer][phInd + 27] = 1  # One Pass
                self.neuralNetworkInputs[nnPlayer][phInd + 27] = 1
                self.neuralNetworkInputs[nnnPlayer][phInd + 27] = 1
            else:
                self.neuralNetworkInputs[nPlayer][phInd + 28] = 1  # Two Passes
                self.neuralNetworkInputs[nnPlayer][phInd + 28] = 1
                self.neuralNetworkInputs[nnnPlayer][phInd + 28] = 1
        else:
            # next player is gaining control.
            self.neuralNetworkInputs[nPlayer][phInd:] = 0
            self.neuralNetworkInputs[nnPlayer][phInd:] = 0
            self.neuralNetworkInputs[nnnPlayer][phInd:] = 0
            self.neuralNetworkInputs[nnnPlayer][phInd + 17] = 1

    def updateNeuralNetworkInputs(self, prevHand, cPlayer):
        self.fillNeuralNetworkHand(cPlayer)
        nPlayer = cPlayer - 1
        if nPlayer == 0:
            nPlayer = 4
        nnPlayer = nPlayer - 1
        if nnPlayer == 0:
            nnPlayer = 4
        nnnPlayer = nnPlayer - 1
        if nnnPlayer == 0:
            nnnPlayer = 4
        nCards = self.currentHands[cPlayer].size
        cardsOfNote = np.intersect1d(prevHand, np.arange(45, 53))
        nPlayerInd = 22 * 13
        nnPlayerInd = nPlayerInd + 27
        nnnPlayerInd = nnPlayerInd + 27
        # next player
        self.neuralNetworkInputs[nPlayer][nPlayerInd : (nPlayerInd + 13)] = 0
        self.neuralNetworkInputs[nPlayer][
            nPlayerInd + nCards - 1
        ] = 1  # number of cards
        # next next player
        self.neuralNetworkInputs[nnPlayer][nnPlayerInd : (nnPlayerInd + 13)] = 0
        self.neuralNetworkInputs[nnPlayer][nnPlayerInd + nCards - 1] = 1
        # next next next player
        self.neuralNetworkInputs[nnnPlayer][nnnPlayerInd : (nnnPlayerInd + 13)] = 0
        self.neuralNetworkInputs[nnnPlayer][nnnPlayerInd + nCards - 1] = 1
        for val in cardsOfNote:
            self.neuralNetworkInputs[nPlayer][nPlayerInd + 13 + (val - 45)] = 1
            self.neuralNetworkInputs[nnPlayer][nnPlayerInd + 13 + (val - 45)] = 1
            self.neuralNetworkInputs[nnnPlayer][nnnPlayerInd + 13 + (val - 45)] = 1
        # prevHand
        phInd = nnnPlayerInd + 27 + 16
        self.neuralNetworkInputs[nPlayer][phInd:] = 0
        self.neuralNetworkInputs[nnPlayer][phInd:] = 0
        self.neuralNetworkInputs[nnnPlayer][phInd:] = 0
        self.neuralNetworkInputs[cPlayer][phInd:] = 0
        nCards = prevHand.size

        if nCards == 2:
            self.neuralNetworkInputs[nPlayer][nPlayerInd + 21] = 1
            self.neuralNetworkInputs[nnPlayer][nnPlayerInd + 21] = 1
            self.neuralNetworkInputs[nnnPlayer][nnnPlayerInd + 21] = 1
            value = int(gameLogic.cardValue(prevHand[1]))
            suit = prevHand[1] % 4
            self.neuralNetworkInputs[nPlayer][phInd + 19] = 1
            self.neuralNetworkInputs[nnPlayer][phInd + 19] = 1
            self.neuralNetworkInputs[nnnPlayer][phInd + 19] = 1
        elif nCards == 3:
            self.neuralNetworkInputs[nPlayer][nPlayerInd + 22] = 1
            self.neuralNetworkInputs[nnPlayer][nnPlayerInd + 22] = 1
            self.neuralNetworkInputs[nnnPlayer][nnnPlayerInd + 22] = 1
            value = int(gameLogic.cardValue(prevHand[2]))
            suit = prevHand[2] % 4
            self.neuralNetworkInputs[nPlayer][phInd + 20] = 1
            self.neuralNetworkInputs[nnPlayer][phInd + 20] = 1
            self.neuralNetworkInputs[nnnPlayer][phInd + 20] = 1
        elif nCards == 4:
            self.neuralNetworkInputs[nPlayer][nPlayerInd + 23] = 1
            self.neuralNetworkInputs[nnPlayer][nnPlayerInd + 23] = 1
            self.neuralNetworkInputs[nnnPlayer][nnnPlayerInd + 23] = 1
            value = int(gameLogic.cardValue(prevHand[3]))
            suit = prevHand[3] % 4
            if gameLogic.isTwoPair(prevHand):
                self.neuralNetworkInputs[nPlayer][phInd + 21] = 1
                self.neuralNetworkInputs[nnPlayer][phInd + 21] = 1
                self.neuralNetworkInputs[nnnPlayer][phInd + 21] = 1
            else:
                self.neuralNetworkInputs[nPlayer][phInd + 22] = 1
                self.neuralNetworkInputs[nnPlayer][phInd + 22] = 1
                self.neuralNetworkInputs[nnnPlayer][phInd + 22] = 1
        elif nCards == 5:
            # import pdb; pdb.set_trace()
            if gameLogic.isStraight(prevHand):
                self.neuralNetworkInputs[nPlayer][nPlayerInd + 24] = 1
                self.neuralNetworkInputs[nnPlayer][nnPlayerInd + 24] = 1
                self.neuralNetworkInputs[nnnPlayer][nnnPlayerInd + 24] = 1
                value = int(gameLogic.cardValue(prevHand[4]))
                suit = prevHand[4] % 4
                self.neuralNetworkInputs[nPlayer][phInd + 23] = 1
                self.neuralNetworkInputs[nnPlayer][phInd + 23] = 1
                self.neuralNetworkInputs[nnnPlayer][phInd + 23] = 1
            if gameLogic.isFlush(prevHand):
                self.neuralNetworkInputs[nPlayer][nPlayerInd + 25] = 1
                self.neuralNetworkInputs[nnPlayer][nnPlayerInd + 25] = 1
                self.neuralNetworkInputs[nnnPlayer][nnnPlayerInd + 25] = 1
                value = int(gameLogic.cardValue(prevHand[4]))
                suit = prevHand[4] % 4
                self.neuralNetworkInputs[nPlayer][phInd + 24] = 1
                self.neuralNetworkInputs[nnPlayer][phInd + 24] = 1
                self.neuralNetworkInputs[nnnPlayer][phInd + 24] = 1
            elif gameLogic.isFullHouse(prevHand):
                self.neuralNetworkInputs[nPlayer][
                    nPlayerInd + 26
                ] = 1  # has played a full house
                self.neuralNetworkInputs[nnPlayer][nnPlayerInd + 26] = 1
                self.neuralNetworkInputs[nnnPlayer][nnnPlayerInd + 26] = 1
                value = int(gameLogic.cardValue(prevHand[2]))
                suit = -1
                self.neuralNetworkInputs[nPlayer][phInd + 25] = 1
                self.neuralNetworkInputs[nnPlayer][phInd + 25] = 1
                self.neuralNetworkInputs[nnnPlayer][phInd + 25] = 1
        else:
            value = int(gameLogic.cardValue(prevHand[0]))
            suit = prevHand[0] % 4
            self.neuralNetworkInputs[nPlayer][phInd + 18] = 1
            self.neuralNetworkInputs[nnPlayer][phInd + 18] = 1
            self.neuralNetworkInputs[nnnPlayer][phInd + 18] = 1
        self.neuralNetworkInputs[nPlayer][phInd + value - 1] = 1
        self.neuralNetworkInputs[nnPlayer][phInd + value - 1] = 1
        self.neuralNetworkInputs[nnnPlayer][phInd + value - 1] = 1
        if suit == 1:
            self.neuralNetworkInputs[nPlayer][phInd + 13] = 1
            self.neuralNetworkInputs[nnPlayer][phInd + 13] = 1
            self.neuralNetworkInputs[nnnPlayer][phInd + 13] = 1
        elif suit == 2:
            self.neuralNetworkInputs[nPlayer][phInd + 14] = 1
            self.neuralNetworkInputs[nnPlayer][phInd + 14] = 1
            self.neuralNetworkInputs[nnnPlayer][phInd + 14] = 1
        elif suit == 3:
            self.neuralNetworkInputs[nPlayer][phInd + 15] = 1
            self.neuralNetworkInputs[nnPlayer][phInd + 15] = 1
            self.neuralNetworkInputs[nnnPlayer][phInd + 15] = 1
        elif suit == 0:
            self.neuralNetworkInputs[nPlayer][phInd + 16] = 1
            self.neuralNetworkInputs[nnPlayer][phInd + 16] = 1
            self.neuralNetworkInputs[nnnPlayer][phInd + 16] = 1
        # general - common to all hands.
        cardsRecord = np.intersect1d(prevHand, np.arange(37, 53))
        endInd = nnnPlayerInd + 27
        for val in cardsRecord:
            self.neuralNetworkInputs[1][endInd + (val - 37)] = 1
            self.neuralNetworkInputs[2][endInd + (val - 37)] = 1
            self.neuralNetworkInputs[3][endInd + (val - 37)] = 1
            self.neuralNetworkInputs[4][endInd + (val - 37)] = 1
        # no passes.
        self.neuralNetworkInputs[nPlayer][phInd + 26] = 1
        self.neuralNetworkInputs[nnPlayer][phInd + 26] = 1
        self.neuralNetworkInputs[nnnPlayer][phInd + 26] = 1
        self.neuralNetworkInputs[nPlayer][phInd + 27 :] = 0
        self.neuralNetworkInputs[nnPlayer][phInd + 27 :] = 0
        self.neuralNetworkInputs[nnnPlayer][phInd + 27 :] = 0

    def getHandsToPlay(self, currHands, option, nCards):
        if nCards == 1:
            handToPlay = np.array([currHands[option]])
        elif nCards == 2:
            handToPlay = currHands[enumerateOptions.inverseTwoCardIndices[option]]
        elif nCards == 3:
            handToPlay = currHands[enumerateOptions.inverseThreeCardIndices[option]]
        # elif nCards == 4:
        # handToPlay = self.currentHands[self.playersGo][enumerateOptions.inverseFourCardIndices[option]]
        elif nCards == 5:
            handToPlay = currHands[enumerateOptions.inverseFiveCardIndices[option]]
        return handToPlay

    def updateGame(self, option, nCards=0):
        self.goCounter += 1
        if option == -1:
            # they pass
            cPlayer = self.playersGo
            self.updateNeuralNetworkPass(cPlayer)
            self.playersGo += 1
            if self.playersGo == 5:
                self.playersGo = 1
            self.passCount += 1
            if self.passCount == 3:
                self.control = 1
                self.passCount = 0
            return
        self.passCount = 0
        handToPlay = self.getHandsToPlay(
            self.currentHands[self.playersGo], option, nCards
        )
        for i in handToPlay:
            self.cardsPlayed[self.playersGo - 1][i - 1] = 1
        self.handsPlayed[self.goIndex] = handPlayed(handToPlay, self.playersGo)
        self.control = 0
        self.goIndex += 1
        self.currentHands[self.playersGo] = np.setdiff1d(
            self.currentHands[self.playersGo], handToPlay
        )
        if self.currentHands[self.playersGo].size == 0:
            self.assignRewards()
            self.gameOver = 1
            return
        self.updateNeuralNetworkInputs(handToPlay, self.playersGo)
        self.playersGo += 1
        if self.playersGo == 5:
            self.playersGo = 1

    def assignRewards(self):
        totCardsLeft = 0
        for i in range(1, 5):
            nC = self.currentHands[i].size
            if ((nC >= 8) and (nC <= 9)) or ((nC == self.whoHas3D) and (nC == 7)):
                nC = 2 * nC
            elif (nC >= 10) and (nC <= 12):
                nC = 3 * nC
            elif nC == 13:
                nC = 4 * nC
            totCardsLeft += nC
        for i in range(1, 5):
            nC = self.currentHands[i].size
            if ((nC >= 8) and (nC <= 9)) or ((nC == self.whoHas3D) and (nC == 7)):
                nC = 2 * nC
            elif (nC >= 10) and (nC <= 12):
                nC = 3 * nC
            elif nC == 13:
                nC = 4 * nC
            if nC == 0:
                winner = i
            else:
                self.rewards[i - 1] = (-4 * nC) + totCardsLeft
        self.rewards[winner - 1] = totCardsLeft + 10

    def randomOption(self):
        cHand = self.currentHands[self.playersGo]
        if self.control == 0:
            prevHand = self.handsPlayed[self.goIndex - 1].hand
            nCards = len(prevHand)
            if nCards > 1:
                handOptions = gameLogic.handsAvailable(cHand)
            if nCards == 1:
                next_player_index = (self.playersGo) % 4 + 1
                next_player_num_cards = len(self.currentHands[next_player_index])
                if next_player_num_cards == 1:  # next player has last card
                    options = enumerateOptions.oneCardOptions(cHand, prevHand, 3)
                else:
                    options = enumerateOptions.oneCardOptions(cHand, prevHand, 1)
            elif nCards == 2:
                options = enumerateOptions.twoCardOptions(handOptions, prevHand, 1)
            elif nCards == 3:
                options = enumerateOptions.threeCardOptions(handOptions, prevHand, 1)
            # elif nCards == 4:
            #     if gameLogic.isFourOfAKind(prevHand):
            #         options = enumerateOptions.fourCardOptions(handOptions, prevHand, 2)
            #     else:
            #         options = enumerateOptions.fourCardOptions(handOptions, prevHand, 1)
            else:
                if gameLogic.isStraight(prevHand):
                    if gameLogic.isFlush(prevHand):
                        options = enumerateOptions.fiveCardOptions(
                            handOptions, prevHand, 5
                        )
                    else:
                        options = enumerateOptions.fiveCardOptions(
                            handOptions, prevHand, 1
                        )
                elif gameLogic.isFlush(prevHand):
                    options = enumerateOptions.fiveCardOptions(handOptions, prevHand, 2)
                elif gameLogic.isKingKong(prevHand):
                    options = enumerateOptions.fiveCardOptions(handOptions, prevHand, 4)
                else:
                    options = enumerateOptions.fiveCardOptions(handOptions, prevHand, 3)
            if isinstance(options, int):
                nOptions = -1
            else:
                nOptions = len(options)
            ind = random.randint(0, nOptions)
            # TODO: handle case where the next player has last card, and we have options, then we cannot pass
            if ind == nOptions or isinstance(options, int):
                return -1  # pass
            else:
                return (options[ind], nCards)
        else:
            # we have control - choose from any option
            handOptions = gameLogic.handsAvailable(cHand)
            # consider the next player has last card
            next_player_index = (self.playersGo) % 4 + 1
            next_player_num_cards = len(self.currentHands[next_player_index])
            if next_player_num_cards == 1:  # next player has last card
                oneCardOptions = enumerateOptions.oneCardOptions(cHand, [], 2)
            else:
                oneCardOptions = enumerateOptions.oneCardOptions(cHand)

            twoCardOptions = enumerateOptions.twoCardOptions(handOptions)
            threeCardOptions = enumerateOptions.threeCardOptions(handOptions)
            # fourCardOptions = enumerateOptions.fourCardOptions(handOptions)
            fiveCardOptions = enumerateOptions.fiveCardOptions(handOptions)
            if isinstance(oneCardOptions, int):
                n1 = 0
            else:
                n1 = len(oneCardOptions)
            if isinstance(twoCardOptions, int):
                n2 = 0
            else:
                n2 = len(twoCardOptions)
            if isinstance(threeCardOptions, int):
                n3 = 0
            else:
                n3 = len(threeCardOptions)
            # if isinstance(fourCardOptions, int):
            # n4 = 0
            # else:
            # n4 = len(fourCardOptions)
            if isinstance(fiveCardOptions, int):
                n5 = 0
            else:
                n5 = len(fiveCardOptions)
            # nTot = n1 + n2 + n3 + n4 + n5
            nTot = n1 + n2 + n3 + n5
            ind = random.randint(0, nTot - 1)
            if ind < n1:
                return (oneCardOptions[ind], 1)
            elif ind < (n1 + n2):
                return (twoCardOptions[ind - n1], 2)
            elif ind < (n1 + n2 + n3):
                return (threeCardOptions[ind - n1 - n2], 3)
            # elif ind < (n1+n2+n3+n4):
            # return (fourCardOptions[ind-n1-n2-n3],4)
            else:
                return (fiveCardOptions[ind - n1 - n2 - n3], 5)

    def returnAvailableActions(self):
        """. This vector is ordered with
        one-card actions in indices 0 − 12,
        two-card actions from 13 − 45,
        three-card actions from 46 − 76,
        four-card actions from 77 − 406,
        five-card actions from 407 − 1693 and then finally
        1694 corresponding to the pass action"""
        currHand = self.currentHands[self.playersGo]
        availableActions = np.zeros((enumerateOptions.nActions[5] + 1,))

        if self.control == 0:
            # allow pass action
            availableActions[enumerateOptions.passInd] = 1

            prevHand = self.handsPlayed[self.goIndex - 1].hand
            nCardsToBeat = len(prevHand)

            if nCardsToBeat > 1:
                handOptions = gameLogic.handsAvailable(currHand)

            if nCardsToBeat == 1:
                next_player_index = (self.playersGo % 4) + 1
                next_player_num_cards = len(self.currentHands[next_player_index])
                if next_player_num_cards == 1:  # next player has last card
                    options = enumerateOptions.oneCardOptions(currHand, prevHand, 3)
                else:
                    options = enumerateOptions.oneCardOptions(currHand, prevHand, 1)
            elif nCardsToBeat == 2:
                options = enumerateOptions.twoCardOptions(handOptions, prevHand, 1)
            elif nCardsToBeat == 3:
                options = enumerateOptions.threeCardOptions(handOptions, prevHand, 1)
            # elif nCardsToBeat == 4:
            # if gameLogic.isFourOfAKind(prevHand):
            # options = enumerateOptions.fourCardOptions(handOptions, prevHand, 2)
            # else:
            # options = enumerateOptions.fourCardOptions(handOptions, prevHand, 1)
            else:
                if gameLogic.isStraight(prevHand):
                    if gameLogic.isFlush(prevHand):
                        options = enumerateOptions.fiveCardOptions(
                            handOptions, prevHand, 5
                        )
                    else:
                        options = enumerateOptions.fiveCardOptions(
                            handOptions, prevHand, 1
                        )
                elif gameLogic.isFlush(prevHand):
                    options = enumerateOptions.fiveCardOptions(handOptions, prevHand, 2)
                elif gameLogic.isKingKong(prevHand):
                    options = enumerateOptions.fiveCardOptions(handOptions, prevHand, 4)
                else:
                    options = enumerateOptions.fiveCardOptions(handOptions, prevHand, 3)

            if isinstance(options, int):  # no options - must pass
                return availableActions
            else:
                # if next player is last card and we have options, we can't pass
                if (nCardsToBeat == 1) and (next_player_num_cards == 1):
                    availableActions[enumerateOptions.passInd] = 0  # not allow pass

            for option in options:
                index = enumerateOptions.getIndex(option, nCardsToBeat)
                availableActions[index] = 1

            return availableActions

        else:  # player has control.
            handOptions = gameLogic.handsAvailable(currHand)
            # consider the next player has last card
            next_player_index = (self.playersGo % 4) + 1
            next_player_num_cards = len(self.currentHands[next_player_index])
            if next_player_num_cards == 1:  # next player has last card
                oneCardOptions = enumerateOptions.oneCardOptions(currHand, [], 2)
            else:
                oneCardOptions = enumerateOptions.oneCardOptions(currHand)
            twoCardOptions = enumerateOptions.twoCardOptions(handOptions)
            threeCardOptions = enumerateOptions.threeCardOptions(handOptions)
            # fourCardOptions = enumerateOptions.fourCardOptions(handOptions)
            fiveCardOptions = enumerateOptions.fiveCardOptions(handOptions)

            for option in oneCardOptions:
                index = enumerateOptions.getIndex(option, 1)
                availableActions[index] = 1
            if not isinstance(twoCardOptions, int):
                for option in twoCardOptions:
                    index = enumerateOptions.getIndex(option, 2)
                    availableActions[index] = 1

            if not isinstance(threeCardOptions, int):
                for option in threeCardOptions:
                    index = enumerateOptions.getIndex(option, 3)
                    availableActions[index] = 1

            # if not isinstance(fourCardOptions, int):
            #     for option in fourCardOptions:
            #         index = enumerateOptions.getIndex(option, 4)
            #         availableActions[index] = 1

            if not isinstance(fiveCardOptions, int):
                for option in fiveCardOptions:
                    index = enumerateOptions.getIndex(option, 5)
                    availableActions[index] = 1

            if self.goIndex == 2:
                # first go - can only play oneCard/ twoCard/ threeCard / fiveCard that has 3D
                availableActions[1:13] = 0  # only 3D is allowed in OneCard options
                # only 3D is allowed in TwoCard options
                # availableActions[13] is (3D,3C)
                # availableActions[14] is (3D,3H)
                # availableActions[15] is (3D,3S)
                # availableActions[16] is (3C,3H), ...
                # Starting from availableActions[16] to availableActions[45] are all pairs without 3D
                availableActions[16:46] = 0
                # only 3D is allowed in ThreeCard options
                # availableActions[46] is (3D,3C,3H)
                # availableActions[47] is (3D,3C,3S)
                # availableActions[48] is (3D,3H,3S), ...
                # Starting from availableActions[49] to availableActions[76] are all triples without 3D
                availableActions[49:77] = 0
                # only 3D is allowed in FiveCard options
                # there are 495 FiveCard actions associated with 3D
                availableActions[407 + 495 :] = 0

            return availableActions

    def step(self, action):
        opt, nC = enumerateOptions.getOptionNC(action)
        self.updateGame(opt, nC)
        if self.gameOver == 0:
            reward = 0
            done = False
            info = None
        else:
            reward = self.rewards
            done = True
            info = {}
            info["numTurns"] = self.goCounter
            info["rewards"] = self.rewards
            # what else is worth monitoring?
            self.reset()
        return reward, done, info

    def getCurrentState(self):
        # action_feats: (n_actions, action_feat_dim)
        availableActions = self.returnAvailableActions().reshape(1, 1695)

        ########## Get action features ##########
        nAvailableActions = int(np.sum(availableActions))
        action_feats = np.zeros(
            (nAvailableActions, 22 * 13)
        )  # shows the remaining hand, each card has 22 features
        availableActionsIndex = np.where(availableActions == 1)[1]
        for i, action in enumerate(availableActionsIndex):
            opt, nC = enumerateOptions.getOptionNC(action)
            if opt == -1:  # pass
                action_feats[i] = self.neuralNetworkInputs[self.playersGo][: 22 * 13]
                continue
            potentialHandsToPlay = self.getHandsToPlay(
                self.currentHands[self.playersGo], opt, nC
            )
            potentialRemainHands = np.setdiff1d(
                self.currentHands[self.playersGo], potentialHandsToPlay
            )
            # potentialHandsToPlay = prevHand
            # potentialRemainHands = self.currentHands[self.playersGo]
            potentialHandOptions = gameLogic.handsAvailable(potentialRemainHands)
            sInd = 0
            for cardID in potentialRemainHands:
                value = potentialHandOptions.cards[cardID].value  # 1-13
                action_feats[i, sInd + int(value) - 1] = 1
                if potentialHandOptions.cards[cardID].inStraight:
                    action_feats[i, sInd + 20] = 1
                sInd += 22

            action_feats[i, np.arange(0, len(potentialRemainHands) * 22, 22) + 13] = (
                potentialHandOptions.suits == 1
            ).astype(int)
            action_feats[i, np.arange(0, len(potentialRemainHands) * 22, 22) + 14] = (
                potentialHandOptions.suits == 2
            ).astype(int)
            action_feats[i, np.arange(0, len(potentialRemainHands) * 22, 22) + 15] = (
                potentialHandOptions.suits == 3
            ).astype(int)
            action_feats[i, np.arange(0, len(potentialRemainHands) * 22, 22) + 16] = (
                potentialHandOptions.suits == 0
            ).astype(int)
            action_feats[
                i, np.arange(0, len(potentialRemainHands) * 22, 22) + 17
            ] = potentialHandOptions.inPair
            action_feats[
                i, np.arange(0, len(potentialRemainHands) * 22, 22) + 18
            ] = potentialHandOptions.inThreeOfAKind
            action_feats[
                i, np.arange(0, len(potentialRemainHands) * 22, 22) + 19
            ] = potentialHandOptions.inFourOfAKind
            action_feats[
                i, np.arange(0, len(potentialRemainHands) * 22, 22) + 21
            ] = potentialHandOptions.inFlush

        action_feats = action_feats.reshape(-1, 22 * 13)

        return (
            self.playersGo,
            self.neuralNetworkInputs[self.playersGo].reshape(1, 412),
            availableActions,
            action_feats,
        )


# now create a vectorized environment
def worker(remote, parent_remote):
    parent_remote.close()
    game = big2Game()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            reward, done, info = game.step(data)
            remote.send((reward, done, info))
        elif cmd == "reset":
            game.reset()
            pGo, cState, availAcs, availActFeats = game.getCurrentState()
            remote.send((pGo, cState))
        elif cmd == "getCurrState":
            pGo, cState, availAcs, availActFeats = game.getCurrentState()
            remote.send((pGo, cState, availAcs, availActFeats))
        elif cmd == "close":
            remote.close()
            break
        else:
            print("Invalid command sent by remote")
            break


class vectorizedBig2Games(object):
    def __init__(self, nGames):
        self.waiting = False
        self.closed = False
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nGames)])
        self.ps = [
            Process(target=worker, args=(work_remote, remote))
            for (work_remote, remote) in zip(self.work_remotes, self.remotes)
        ]

        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        rewards, dones, infos = zip(*results)
        return rewards, dones, infos

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def currStates_async(self):
        for remote in self.remotes:
            remote.send(("getCurrState", None))
        self.waiting = True

    def currStates_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        pGos, currStates, currAvailAcs, currAvailActFeats = zip(*results)
        return (
            np.stack(pGos),
            np.stack(currStates),
            np.stack(currAvailAcs),
            [currAvailActFeat for currAvailActFeat in currAvailActFeats],
        )

    def getCurrStates(self):
        self.currStates_async()
        return self.currStates_wait()

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.ps:
            p.join()
        self.closed = True
