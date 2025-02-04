#enumerate options
import pickle
import numpy as np
import big2.gameLogic as gameLogic
import os


nActions = np.array([13,33,31,330,1287,1694])
nAcSum = np.cumsum(nActions[:-1])

cur_dir = os.path.dirname(os.path.abspath(__file__))
with open(cur_dir+'/actionIndices.pkl','rb') as f:  # Python 3: open(..., 'rb')
    twoCardIndices, threeCardIndices, fourCardIndices, fiveCardIndices, inverseTwoCardIndices, inverseThreeCardIndices, inverseFourCardIndices, inverseFiveCardIndices = pickle.load(f)

passInd = nActions[-1]

def getIndex(option, nCards):
    if nCards==0: #pass
        return passInd
    sInd = 0
    for i in range(nCards-1):
        sInd += nActions[i]
    return sInd + option

def getOptionNC(ind):
    if ind == passInd:
        return -1, 0
    if ind < nAcSum[0]:
        return ind, 1
    elif ind < nAcSum[1]:
        return ind - nAcSum[0], 2
    elif ind < nAcSum[2]:
        return ind - nAcSum[1], 3
    elif ind < nAcSum[3]:
        return ind - nAcSum[2], 4
    else:
        return ind - nAcSum[3], 5

def fiveCardOptions(handOptions, prevHand=[],prevType=0):
    #prevType = 0 - no hand, you have control and can play any 5 card
    #         = 1 - straight
    #         = 2 - flush
    #         = 3 - full house
    #         = 4 - King Kong
    #         = 5 - straight flush
        
    validInds = np.zeros((nActions[4],),dtype=int)
    c = 0
    cardInds = np.zeros((5,),dtype=int) #reuse
    
    #first deal with straights
    if prevType == 2 or prevType == 3 or prevType == 4 or prevType == 5:
        pass
    else:
        # handOptions.straights is list of list of card indices that are in a straight
        if len(handOptions.straights) > 0:
            for straight in handOptions.straights:
                nC = straight.size
                for i1 in range(nC-4): #first index of hand
                    val1 = handOptions.cards[straight[i1]].value
                    cardInds[0] = handOptions.cards[straight[i1]].indexInHand
                    # Does not allow JQKA2, QKA23, KA234
                    # If val1 is J or Q or K, then it is invalid
                    if (val1 == 9) or (val1 == 10) or (val1 == 11):
                        continue
                    for i2 in range(i1+1,nC-3):
                        val2 = handOptions.cards[straight[i2]].value
                        if val1 == val2:
                            continue
                        # if (val1 is 5 and val2 is A) or (val1 is 6 and val2 is 2), then it is still valid
                        if (val2 > val1 + 1) and (not ((val1 == 3) and (val2 == 12))) and (not ((val1 == 4) and (val2 == 13))):
                            break
                        cardInds[1] = handOptions.cards[straight[i2]].indexInHand
                        for i3 in range(i2+1,nC-2):
                            val3 = handOptions.cards[straight[i3]].value
                            if val3 == val2:
                                continue
                            if (val3 > val2 + 1) and (not ((val2 == 3) and (val3 == 12))) and (not ((val2 == 4) and (val3 == 13))):
                                break
                            cardInds[2] = handOptions.cards[straight[i3]].indexInHand
                            for i4 in range(i3+1,nC-1):
                                val4 = handOptions.cards[straight[i4]].value
                                if val4 == val3:
                                    continue
                                if (val4 > val3 + 1) and (not ((val3 == 3) and (val4 == 12))) and (not ((val3 == 4) and (val4 == 13))):
                                    break
                                cardInds[3] = handOptions.cards[straight[i4]].indexInHand
                                for i5 in range(i4+1, nC):
                                    val5 = handOptions.cards[straight[i5]].value
                                    if val5 == val4:
                                        continue
                                    if (val5 > val4 + 1) and (not ((val4 == 3) and (val5 == 12))) and (not ((val4 == 4) and (val5 == 13))):
                                        break
                                    #import pdb; pdb.set_trace()
                                    cardInds[4] = handOptions.cards[straight[i5]].indexInHand
                                    if prevType == 1:
                                        if handOptions.cHand[cardInds[4]] < prevHand[4]:
                                            continue
                                    if c == 1287: 
                                        print("==>> handOptions.straights: ", handOptions.straights)
                                        print("==>> handOptions.cHand: ", handOptions.cHand)
                                        
                                    validInds[c] = fiveCardIndices[cardInds[0]][cardInds[1]][cardInds[2]][cardInds[3]][cardInds[4]]
                                    c += 1
    
    #now deal with flushes (easier)
    if len(handOptions.flushes) > 0:
        for flush in handOptions.flushes:
            #all combinations of flush are allowable
            nC = flush.size
            for i1 in range(nC-4):
                cardInds[0] = handOptions.cards[flush[i1]].indexInHand
                for i2 in range(i1+1,nC-3):
                    cardInds[1] = handOptions.cards[flush[i2]].indexInHand
                    for i3 in range(i2+1,nC-2):
                        cardInds[2] = handOptions.cards[flush[i3]].indexInHand
                        for i4 in range(i3+1,nC-1):
                            cardInds[3] = handOptions.cards[flush[i4]].indexInHand
                            for i5 in range(i4+1,nC):
                                cardInds[4] = handOptions.cards[flush[i5]].indexInHand
                                if prevType == 2:
                                    if prevHand[4] > handOptions.cHand[cardInds[4]]:  # if prevHand is higher, then we can still beat it if we were straight flush
                                        handBeingPlayed = handOptions.cHand[cardInds]
                                        if gameLogic.isStraight(handBeingPlayed): #its a straight flush so wins
                                            pass
                                        else:
                                            continue
                                if (prevType == 3) or (prevType == 4): #needs to be a straight flush to beat
                                    handBeingPlayed = handOptions.cHand[cardInds]
                                    if gameLogic.isStraight(handBeingPlayed): #its a straight flush so wins
                                        pass
                                    else:
                                        continue
                                if (prevType == 5): 
                                    handBeingPlayed = handOptions.cHand[cardInds]
                                    if gameLogic.isStraight(handBeingPlayed):
                                        if (prevHand[4] > handOptions.cHand[cardInds[4]]):
                                            continue  # we can't beat it with straight flush becasue we are weaker straight flush
                                        else:
                                            pass  # we beat it with bigger straight flush
                                    else:
                                        continue
                                validInds[c] = fiveCardIndices[cardInds[0]][cardInds[1]][cardInds[2]][cardInds[3]][cardInds[4]]
                                c += 1
    
    #now deal with full houses
    if prevType <= 3:
        if prevType == 3:
            threeVal = gameLogic.cardValue(prevHand[2])
        nPairs = handOptions.nPairs
        nThree = handOptions.nThreeOfAKinds
        if nPairs > 0 and nThree > 0:
            for pair in handOptions.pairs:
                pVal = handOptions.cards[pair[0]].value
                for three in handOptions.threeOfAKinds:
                    tVal = handOptions.cards[three[0]].value
                    if tVal == pVal:
                        continue
                    if prevType == 3:
                        if threeVal > tVal:  # we lose since we are lower three of a kind
                            continue
                    if pVal > tVal:
                        cardInds[0] = handOptions.cards[three[0]].indexInHand
                        cardInds[1] = handOptions.cards[three[1]].indexInHand
                        cardInds[2] = handOptions.cards[three[2]].indexInHand
                        cardInds[3] = handOptions.cards[pair[0]].indexInHand
                        cardInds[4] = handOptions.cards[pair[1]].indexInHand
                    else:
                        cardInds[0] = handOptions.cards[pair[0]].indexInHand
                        cardInds[1] = handOptions.cards[pair[1]].indexInHand
                        cardInds[2] = handOptions.cards[three[0]].indexInHand
                        cardInds[3] = handOptions.cards[three[1]].indexInHand
                        cardInds[4] = handOptions.cards[three[2]].indexInHand
                    validInds[c] = fiveCardIndices[cardInds[0]][cardInds[1]][cardInds[2]][cardInds[3]][cardInds[4]]
                    c += 1
    # now deal with king kong
    if prevType <= 4:
        if prevType == 4:
            fourVal = gameLogic.cardValue(prevHand[3])
        if (len(handOptions.fourOfAKinds) > 0) and (handOptions.handLength > 4):
            for four in handOptions.fourOfAKinds:
                fVal = handOptions.cards[four[0]].value
                if prevType == 4:
                    if fourVal > fVal:
                        continue # we lose since we are lower four of a kind
                for single in handOptions.cHand:
                    oVal = handOptions.cards[single].value  # the one card we are adding
                    if oVal == fVal:  # actually impossible
                        continue
                    if oVal > fVal:
                        cardInds[0] = handOptions.cards[four[0]].indexInHand
                        cardInds[1] = handOptions.cards[four[1]].indexInHand
                        cardInds[2] = handOptions.cards[four[2]].indexInHand
                        cardInds[3] = handOptions.cards[four[3]].indexInHand
                        cardInds[4] = handOptions.cards[single].indexInHand
                    else:
                        cardInds[0] = handOptions.cards[single].indexInHand
                        cardInds[1] = handOptions.cards[four[0]].indexInHand
                        cardInds[2] = handOptions.cards[four[1]].indexInHand
                        cardInds[3] = handOptions.cards[four[2]].indexInHand
                        cardInds[4] = handOptions.cards[four[3]].indexInHand
                    validInds[c] = fiveCardIndices[cardInds[0]][cardInds[1]][cardInds[2]][cardInds[3]][cardInds[4]]
                    c += 1

    if c > 0:
        return validInds[0:c]
    else:
        return -1

    

def fourCardOptions(handOptions, prevHand = [], prevType = 0):
    #prevType: 1 - pair, 2 - fourofakind    
    validInds = np.zeros((nActions[3],),dtype=int)
    c = 0
    cardInds = np.zeros((4,),dtype=int) #reuse
    
    #four of a kinds
    if len(handOptions.fourOfAKinds) > 0:
        for four in handOptions.fourOfAKinds:
            cardInds[0] = handOptions.cards[four[0]].indexInHand
            cardInds[1] = handOptions.cards[four[1]].indexInHand
            cardInds[2] = handOptions.cards[four[2]].indexInHand
            cardInds[3] = handOptions.cards[four[3]].indexInHand
            if prevType == 2:
                if handOptions.cHand[cardInds[0]] < prevHand[3]:
                    continue
            validInds[c] = fourCardIndices[cardInds[0]][cardInds[1]][cardInds[2]][cardInds[3]]
            c += 1
    #two pairs
    if prevType == 2:
        pass
    else:
        if handOptions.nDistinctPairs >= 2:
            nPairs = handOptions.nPairs
            for p1 in range(nPairs-1):
                p1Val = handOptions.cards[handOptions.pairs[p1][0]].value
                for p2 in range(p1+1,nPairs):
                    p2Val = handOptions.cards[handOptions.pairs[p2][0]].value
                    if p1Val == p2Val:
                        continue
                    cardInds[0] = handOptions.cards[handOptions.pairs[p1][0]].indexInHand
                    cardInds[1] = handOptions.cards[handOptions.pairs[p1][1]].indexInHand
                    cardInds[2] = handOptions.cards[handOptions.pairs[p2][0]].indexInHand
                    cardInds[3] = handOptions.cards[handOptions.pairs[p2][1]].indexInHand
                    if prevType == 1:
                        if handOptions.cHand[cardInds[3]] < prevHand[3]:
                            continue
                    validInds[c] = fourCardIndices[cardInds[0]][cardInds[1]][cardInds[2]][cardInds[3]]
                    c += 1
    if c > 0:
        return validInds[0:c]
    else:
        return -1 #no four card hands

def threeCardOptions(handOptions, prevHand = [], prevType = 0):
    #prevType = 1 - played three of a kind
    validInds = np.zeros((nActions[2],), dtype=int)
    c = 0
    cardInds = np.zeros((3,), dtype=int) #reuse
    
    if handOptions.nThreeOfAKinds > 0:
        for three in handOptions.threeOfAKinds:
            cardInds[0] = handOptions.cards[three[0]].indexInHand
            cardInds[1] = handOptions.cards[three[1]].indexInHand
            cardInds[2] = handOptions.cards[three[2]].indexInHand
            if prevType == 1:
                if handOptions.cHand[cardInds[2]] < prevHand[2]:
                    continue
            validInds[c] = threeCardIndices[cardInds[0]][cardInds[1]][cardInds[2]]
            c += 1
    if c > 0:
        return validInds[0:c]
    else:
        return -1

def twoCardOptions(handOptions, prevHand = [], prevType = 0):
    #prevType = 1 - played a pair
    validInds = np.zeros((nActions[1],), dtype=int)
    c = 0
    cardInds = np.zeros((2,), dtype=int)
    
    if handOptions.nPairs > 0:
        # handOptions.pairs: list of pairs of cardID (cardID is a number from 1 to 52)
        for pair in handOptions.pairs:
            # handOptions.cards is a indexInHand2card dictionary
            cardInds[0] = handOptions.cards[pair[0]].indexInHand
            cardInds[1] = handOptions.cards[pair[1]].indexInHand
            if prevType == 1:
                if handOptions.cHand[cardInds[1]] < prevHand[1]:
                    continue
            validInds[c] = twoCardIndices[cardInds[0]][cardInds[1]]
            c += 1
    if c > 0:
        return validInds[0:c]
    else:
        return -1
    
def oneCardOptions(hand, prevHand = [], prevType = 0):
    #prevType = 1 - played a oneCard
    #prevType = 2 next player has last card and in control
    #prevType = 3 next player has last card and someone has played a oneCard
    nCards = len(hand)
    validInds = np.zeros((nCards,), dtype=int)
    c = 0
    # First consider the case where the next player has the last card
    if (prevType == 2) or (prevType == 3):
        largest = max(hand)
        if prevType == 3:
            if prevHand > largest:
                return -1
        return np.array([nCards-1], dtype=int)  # need to play the largest card
    for i in range(nCards):
        if (prevType == 1):
            if prevHand > hand[i]:
                continue
        validInds[c] = i
        c += 1
    if c > 0:
        return validInds[0:c]
    else:
        return -1