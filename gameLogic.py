import numpy as np
import itertools


def shuffle(array):
    i = 0
    j = 0
    temp = 0

    for i in range(array.size - 1, 0, -1):
        j = int(np.floor(np.random.random() * (i + 1)))
        temp = array[i]
        array[i] = array[j]
        array[j] = temp
    return array


def isPair(hand):
    if hand.size != 2:
        return 0
    if np.ceil(hand[0] / 4) == np.ceil(hand[1] / 4):
        return 1
    else:
        return 0


def isThreeOfAKind(hand):
    if hand.size != 3:
        return 0
    if (np.ceil(hand[0] / 4) == np.ceil(hand[1] / 4)) and (
        np.ceil(hand[1] / 4) == np.ceil(hand[2] / 4)
    ):
        return 1
    else:
        return 0


def isFourOfAKind(hand):
    if hand.size != 4:
        return 0
    if (
        (np.ceil(hand[0] / 4) == np.ceil(hand[1] / 4))
        and (np.ceil(hand[1] / 4) == np.ceil(hand[2] / 4))
        and (np.ceil(hand[2] / 4) == np.ceil(hand[3] / 4))
    ):
        return 1
    else:
        return 0


def isTwoPair(hand):
    if hand.size != 4:
        return 0
    if isFourOfAKind(hand):
        return 0
    hand.sort()
    if isPair(hand[0:2]) and isPair(hand[2:]):
        return 1
    else:
        return 0


def isStraightFlush(hand):
    if hand.size != 5:
        return 0
    hand.sort()
    if (
        (hand[0] + 4 == hand[1])
        and (hand[1] + 4 == hand[2])
        and (hand[2] + 4 == hand[3])
        and (hand[3] + 4 == hand[4])
    ):
        return 1
    else:
        return 0


def isStraight(hand):
    if hand.size != 5:
        return 0
    hand.sort()
    # Don't allow JQKA2, QKA23, KA234
    # Allow A2345 and 23456

    # number 1 means 3, 2 means 4, 3 means 5, 4 means 6, 5 means 7, 6 means 8, 7 means 9, 8 means 10, 9 means J, 10 means Q, 11 means K, 12 means A, 13 means 2
    number1 = np.ceil(hand[0] / 4)
    number2 = np.ceil(hand[1] / 4)
    number3 = np.ceil(hand[2] / 4)
    number4 = np.ceil(hand[3] / 4)
    number5 = np.ceil(hand[4] / 4)
    # Allow A2345
    if (
        (number1 == 1)
        and (number2 == 2)
        and (number3 == 3)
        and (number4 == 12)
        and (number5 == 13)
    ):
        return 1
    # Allow 23456
    if (
        (number1 == 1)
        and (number2 == 2)
        and (number3 == 3)
        and (number4 == 4)
        and (number5 == 13)
    ):
        return 1
    if (
        (number1 + 1 == number2)
        and (number2 + 1 == number3)
        and (number3 + 1 == number4)
        and (number4 + 1 == number5)
    ):
        if (
            (number1 == 9) or (number1 == 10) or (number1 == 11)
        ):  # Don't allow JQKA2, QKA23, KA234, 9 is J
            return 0
        return 1
    else:
        return 0


def isFlush(hand):
    if hand.size != 5:
        return 0
    if (
        (hand[0] % 4 == hand[1] % 4)
        and (hand[1] % 4 == hand[2] % 4)
        and (hand[2] % 4 == hand[3] % 4)
        and (hand[3] % 4 == hand[4] % 4)
    ):
        return 1
    else:
        return 0


# returns the value of the 3 card part
def isFullHouse(hand):
    if hand.size != 5:
        return (False,)
    hand.sort()
    if isPair(hand[0:2]) and isThreeOfAKind(hand[2:]):
        return (True, np.ceil(hand[3] / 4))
    elif isThreeOfAKind(hand[0:3]) and isPair(hand[3:]):
        return (True, np.ceil(hand[0] / 4))
    else:
        return (False,)


def isKingKong(hand):
    if hand.size != 5:
        return 0
    hand.sort()
    if isFourOfAKind(hand[0:4]) or isFourOfAKind(hand[1:]):
        return 1
    else:
        return 0


def isRealHand(hand):
    if (hand.size > 5) or (hand.size < 1):
        return 0
    if hand.size == 1:
        return 1
    if hand.size == 2:
        if isPair(hand):
            return 1
        else:
            return 0
    if hand.size == 3:
        if isThreeOfAKind(hand):
            return 1
        else:
            return 0
    if hand.size == 4:
        if isTwoPair(hand):
            return 1
        elif isFourOfAKind(hand):
            return 1
        else:
            return 0
    if hand.size == 5:
        if isStraight(hand):
            return 1
        elif isFlush(hand):
            return 1
        elif isFullHouse(hand):
            return 1
        else:
            return 0


def validatePlayedHand(hand, prevHand, control):
    if not isRealHand(hand):
        return 0
    if control == 1:
        return 1  # can play any real hand with control
    if hand.size != prevHand.size:
        return 0  # must be same size if not in control

    hand.sort()
    prevHand.sort()

    if hand.size == 1:
        if hand[0] > prevHand[0]:
            return 1
        else:
            return 0
    elif hand.size == 2:
        if not isPair(hand):
            return 0
        else:
            if hand[1] > prevHand[1]:
                return 1
            else:
                return 0
    elif hand.size == 3:
        if not isThreeOfAKind(hand):
            return 0
        else:
            if hand[2] > prevHand[2]:
                return 1
            else:
                return 0
    elif hand.size == 4:
        if isFourOfAKind(hand):
            if not isFourOfAKind(prevHand):
                return 1
            else:
                if hand[3] > prevHand[3]:
                    return 1
                else:
                    return 0
        if isTwoPair(hand):
            if isFourOfAKind(prevHand):
                return 0
            else:
                if hand[3] > prevHand[3]:
                    return 1
                else:
                    return 0
        return 0
    elif hand.size == 5:
        if isStraightFlush(hand):
            if not isStraightFlush(prevHand):
                return 1
            else:
                if hand[4] > prevHand[4]:
                    return 1
                else:
                    return 0

        fh = isFullHouse(hand)
        if fh[0] == True:
            if isStraightFlush(prevHand):
                return 0
            fhph = isFullHouse(prevHand)
            if fhph[0] == False:
                return 1
            else:
                if fh[1] > fhph[1]:
                    return 1
                else:
                    return 0

        if isFlush(hand):
            if isFullHouse(prevHand)[0]:
                return 0
            elif isStraightFlush(prevHand):
                return 0
            if not isFlush(prevHand):
                return 1
            else:
                if hand[4] > prevHand[4]:
                    return 1
                else:
                    return 0

        if isStraight(hand):
            if isFullHouse(prevHand)[0]:
                return 0
            elif isFlush(prevHand):
                return 0
            elif isStraightFlush(prevHand):
                return 0

            if hand[4] > prevHand[4]:
                return 1
            else:
                return 0


# function to convert hand in text form into number form.
def convertHand(hand):
    # takes a list in the form ["3H","KD",...] etc and converts it into numbers
    output = np.zeros(len(hand))
    counter = 0
    for card in hand:
        if card[0] == "2":
            base = 13
        elif card[0] == "A":
            base = 12
        elif card[0] == "K":
            base = 11
        elif card[0] == "Q":
            base = 10
        elif card[0] == "J":
            base = 9
        elif card[0] == "1":
            base = 8
            card = card.replace("0", "")
        else:
            base = int(card[0]) - 2

        if card[1] == "D":
            suit = 1
        elif card[1] == "C":
            suit = 2
        elif card[1] == "H":
            suit = 3
        elif card[1] == "S":
            suit = 4

        output[counter] = int((base - 1) * 4 + suit)
        counter += 1
    return output


# we need a function that evaluates an initial hand and evaluates all of the hands which are available.
# so have a vector of 2 card hands, 3 card hands, etc. We should then have a function which updates all of the available hands
# when a particular hand is played.


def cardValue(num):
    return np.ceil(num / 4)


class card:
    def __init__(self, number, i):
        # self.suit = number % 4  # 1 - Diamond, 2 - Club, 3- Heart, 0 - Spade
        self.value = np.ceil(number / 4)  # from 1 to 13.
        self.indexInHand = i  # index within current hand (from 0 to 12)
        self.inStraight = 0
        self.straightIndex = -1  # index of which straight this card is in.

    def __repr__(self):
        if self.value < 8:
            string1 = str(self.value + 2)
            string1 = string1[0]
        elif self.value == 8:
            string1 = "10"
        elif self.value == 9:
            string1 = "J"
        elif self.value == 10:
            string1 = "Q"
        elif self.value == 11:
            string1 = "K"
        elif self.value == 12:
            string1 = "A"
        elif self.value == 13:
            string1 = "2"
        if self.suit == 1:
            string2 = "D"
        elif self.suit == 2:
            string2 = "C"
        elif self.suit == 3:
            string2 = "H"
        else:
            string2 = "S"
        cardString = string1 + string2
        return "<card. %s, inStraight: %d>" % (
            cardString,
            self.inStraight,
        )


class handsAvailable:
    def __init__(self, currentHand, nC=0):
        self.cHand = np.sort(currentHand).astype(int)
        self.handLength = currentHand.size
        self.cards = {}
        for i in range(self.cHand.size):
            self.cards[self.cHand[i]] = card(self.cHand[i], i)
        self.values = np.ceil(self.cHand / 4).astype(int)
        self.suits = self.cHand % 4
        self.flushes = []
        self.pairs = []
        self.threeOfAKinds = []
        self.fourOfAKinds = []
        self.straights = []
        self.nPairs = 0
        self.nThreeOfAKinds = 0
        self.nDistinctPairs = 0

        self.unique_values, self.unique_values_counts = np.unique(
            self.values, return_counts=True
        )
        if nC == 2:
            self.fillPairs()
        elif nC == 3:
            self.fillThreeOfAKinds()
        elif nC == 4:
            self.fillFourOfAKinds()
            self.fillPairs()
        else:
            self.fillPairs()
            self.fillSuits()
            self.fillStraights()
            self.fillThreeOfAKinds()
            self.fillFourOfAKinds()

    def fillSuits(self):
        # self.diamonds = np.zeros((self.handLength,))
        # self.clubs = np.zeros((self.handLength,))
        # self.hearts = np.zeros((self.handLength,))
        # self.spades = np.zeros((self.handLength,))
        # dc = 0
        # cc = 0
        # hc = 0
        # sc = 0
        # for i in range(self.handLength):
        #     val = self.cHand[i] % 4
        #     if val == 1:
        #         self.diamonds[dc] = self.cHand[i]
        #         dc += 1
        #     elif val == 2:
        #         self.clubs[cc] = self.cHand[i]
        #         cc += 1
        #     elif val == 3:
        #         self.hearts[hc] = self.cHand[i]
        #         hc += 1
        #     else:
        #         self.spades[sc] = self.cHand[i]
        #         sc += 1
        # self.diamonds = self.diamonds[0:dc]
        # self.clubs = self.clubs[0:cc]
        # self.hearts = self.hearts[0:hc]
        # self.spades = self.spades[0:sc]
        # if self.diamonds.size >= 5:
        #     self.flushes.append(self.diamonds)
        # if self.clubs.size >= 5:
        #     self.flushes.append(self.clubs)
        # if self.hearts.size >= 5:
        #     self.flushes.append(self.hearts)
        # if self.spades.size >= 5:
        #     self.flushes.append(self.spades)
        # for i in range(len(self.flushes)):
        #     flushes = self.flushes[i]
        #     for j in range(flushes.size):
        #         self.cards[flushes[j]].inFlush = 1

        ##############################################
        suits, counts = np.unique(self.suits, return_counts=True)
        suits_in_flush = suits[counts >= 5]
        self.inFlush = np.isin(self.suits, suits_in_flush).astype(int)
        for suit in suits_in_flush:
            self.flushes.append(self.cHand[self.suits == suit])

    def fillStraights(self):
        """Finds all of the straights in the current hand."""
        # Also need to add A2345 and 23456
        exist_2 = False
        exist_3 = False
        exist_4 = False
        exist_5 = False
        exist_6 = False
        exist_A = False
        ind_weak_A = self.cHand.size
        ind_weak_2 = self.cHand.size
        for i, cNum in enumerate(self.cHand):
            if self.cards[cNum].value == 13:
                exist_2 = True
                if ind_weak_2 == self.cHand.size:
                    ind_weak_2 = i
            elif self.cards[cNum].value == 1:
                exist_3 = True
            elif self.cards[cNum].value == 2:
                exist_4 = True
            elif self.cards[cNum].value == 3:
                exist_5 = True
                ind_strong_5 = i  # the index of the strongest 5
            elif self.cards[cNum].value == 4:
                exist_6 = True
                ind_strong_6 = i  # the index of the strongest 6
            elif self.cards[cNum].value == 12:
                exist_A = True
                if ind_weak_A == self.cHand.size:
                    ind_weak_A = i  # the index of the weakest A

        # Consider the straights other than A2345 and 23456 first, i.e. only consider 3-A

        streak = 0
        cInd = 0
        sInd = 0

        while cInd < ind_weak_2 - 1:
            cVal = self.cards[self.cHand[cInd]].value
            nVal = self.cards[self.cHand[cInd + 1]].value
            if nVal == cVal + 1:
                streak += 1
                cInd += 1
            elif nVal == cVal:
                cInd += 1
            else:
                if streak >= 4:
                    self.straights.append(self.cHand[sInd : cInd + 1])
                streak = 0
                cInd = cInd + 1
                sInd = cInd
        if streak >= 4:
            self.straights.append(self.cHand[sInd:ind_weak_2])

        # Now consider A2345 and 23456
        # if 3 is in hand, then self.cHand[0] is always the index of the weakest 3
        if (
            exist_2 and exist_3 and exist_4 and exist_5 and exist_6
        ):  # 23456, we add 34562
            straight_23456 = np.concatenate(
                (self.cHand[0 : ind_strong_6 + 1], self.cHand[ind_weak_2:])
            )
            self.straights.append(straight_23456)  # the 23456 straights, we add 34562
        if (
            exist_A and exist_2 and exist_3 and exist_4 and exist_5
        ):  # A2345, we add 345A2
            straight_A2345 = np.concatenate(
                (self.cHand[0 : ind_strong_5 + 1], self.cHand[ind_weak_A:])
            )
            self.straights.append(straight_A2345)  # the 2345A straights
        for straight in self.straights:
            for cNum in straight:
                self.cards[cNum].inStraight = 1
                self.cards[cNum].straightIndex = i

    def fillPairs(self):
        # cVal = -1
        # nDistinct = 0
        # for i in range(self.handLength - 1):
        #     for j in range(i + 1, i + 4):
        #         if j >= self.handLength:
        #             continue
        #         if isPair(np.array([self.cHand[i], self.cHand[j]])):
        #             nVal = cardValue(self.cHand[i])
        #             if nVal != cVal:
        #                 nDistinct += 1
        #                 cVal = nVal
        #             self.pairs.append([self.cHand[i], self.cHand[j]])
        #             self.nPairs += 1
        #             self.nDistinctPairs = nDistinct
        #             self.cards[self.cHand[i]].inPair = 1
        #             self.cards[self.cHand[j]].inPair = 1
        ##########################
        values_in_pair = self.unique_values[self.unique_values_counts >= 2]
        self.inPair = np.isin(self.values, values_in_pair).astype(int)
        self.nDistinctPairs = len(values_in_pair)
        self.pairs = [
            [pair1, pair2]
            for value in values_in_pair
            for pair1, pair2 in itertools.combinations(
                self.cHand[self.values == value], 2
            )
        ]
        self.nPairs = len(self.pairs)

    def fillThreeOfAKinds(self):
        # for i in range(self.handLength - 2):
        #     for j in range(i + 1, i + 3):
        #         if (j + 1) >= self.handLength:
        #             continue
        #         if isThreeOfAKind(
        #             np.array([self.cHand[i], self.cHand[j], self.cHand[j + 1]])
        #         ):
        #             self.threeOfAKinds.append(
        #                 [self.cHand[i], self.cHand[j], self.cHand[j + 1]]
        #             )
        #             self.nThreeOfAKinds += 1
        #             self.cards[self.cHand[i]].inThreeOfAKind = 1
        #             self.cards[self.cHand[j]].inThreeOfAKind = 1
        #             self.cards[self.cHand[j + 1]].inThreeOfAKind = 1
        # ############################
        values_in_three_of_a_kind = self.unique_values[self.unique_values_counts >= 3]
        self.inThreeOfAKind = np.isin(self.values, values_in_three_of_a_kind).astype(
            int
        )
        self.nThreeOfAKinds = len(values_in_three_of_a_kind)
        self.threeOfAKinds = [
            [three_of_a_kind1, three_of_a_kind2, three_of_a_kind3]
            for value in values_in_three_of_a_kind
            for three_of_a_kind1, three_of_a_kind2, three_of_a_kind3 in itertools.combinations(
                self.cHand[self.values == value], 3
            )
        ]

    def fillFourOfAKinds(self):
        # for i in range(self.handLength - 3):
        #     if self.cards[self.cHand[i]].suit == 1:
        #         if np.ceil(self.cHand[i] / 4) == np.ceil(self.cHand[i + 1] / 4):
        #             if np.ceil(self.cHand[i] / 4) == np.ceil(self.cHand[i + 2] / 4):
        #                 if np.ceil(self.cHand[i] / 4) == np.ceil(self.cHand[i + 3] / 4):
        #                     self.fourOfAKinds.append(
        #                         [
        #                             self.cHand[i],
        #                             self.cHand[i + 1],
        #                             self.cHand[i + 2],
        #                             self.cHand[i + 3],
        #                         ]
        #                     )
        #                     self.cards[self.cHand[i]].inFourOfAKind = 1
        #                     self.cards[self.cHand[i + 1]].inFourOfAKind = 1
        #                     self.cards[self.cHand[i + 2]].inFourOfAKind = 1
        #                     self.cards[self.cHand[i + 3]].inFourOfAKind = 1
        # ################################
        values_in_four_of_a_kind = self.unique_values[self.unique_values_counts >= 4]
        if len(values_in_four_of_a_kind) == 0:
            self.inFourOfAKind = np.zeros(self.handLength, dtype=int)
            self.fourOfAKinds = []
        else:
            self.inFourOfAKind = np.isin(self.values, values_in_four_of_a_kind).astype(int)
            self.fourOfAKinds = [
                self.cHand[self.values == value]
                for value in values_in_four_of_a_kind
            ]
