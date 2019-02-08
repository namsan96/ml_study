import numpy as np
import copy

# Blackjack Environment

class Card:
    def __init__(self, s, r):
        self.suit = s  # S(Spades), C(Clubs), D(Diamonds), H(Hearts)
        self.rank = r  # A, 2, 3, .., T(10), J, Q, K

        if s == 'S' or s == 'C':
            self.color = 'Black'
        else:
            self.color = 'Red'
        if r == 'A':
            self.number = 1
        elif r == 'T':
            self.number = 10
        elif r == 'J':
            self.number = 11
        elif r == 'Q':
            self.number = 12
        elif r == 'K':
            self.number = 13
        else:
            self.number = ord(r) - ord('0')
        if self.number >= 10:
            self.BJnumber = 10
        elif self.number == 1:
            self.BJnumber = 11
        else:
            self.BJnumber = self.number

class CardDeck:
    def __init__(self, n):
        self.n_deck = n   # number of card decks
        self.dCount = np.zeros(12, dtype=int) # number of cards having dCount idx figure in face
        self.make_multideck(self.n_deck)
        self.reshuffle_factor = 0.85
        self.reshuffle()


    rank_set = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
    suit_set = ['S', 'D', 'C', 'H']
    def make_suit(self, s):
        for r in self.rank_set:
            c = Card(s, r)
            self.deck.append(c)

    def make_52deck(self):
        for s in self.suit_set:
            self.make_suit(s)

    def make_multideck(self, n_deck):
        self.deck = []
        for i in range(n_deck):
            self.make_52deck()
        self.reshuffle()

    def reset_count(self):
        N = self.n_deck*4
        self.dCount[0:2].fill(0)
        self.dCount[2:10].fill(N)
        self.dCount[10] = N*4
        self.dCount[11] = N

    def reshuffle(self):
        np.random.shuffle(self.deck)
        self.next_card = 0
        self.reset_count()

    def need_shuffle(self):
        return self.next_card >= self.n_deck * 52 * self.reshuffle_factor

    def remaining_cards(self):
        return self.n_deck*52 - self.next_card

    def get_next_card(self):
        c = self.deck[self.next_card]
        self.next_card += 1
        self.dCount[c.BJnumber] -= 1
        if self.remaining_cards() == 0:
            self.reshuffle()  # no remaining card, enforce reshuffle
        return c

    def get_next_card_BJ(self):
        return self.get_next_card().BJnumber

    def peep(self):
        return self.dCount[2:12] / self.remaining_cards()

    def antithetic(self):
        N = self.n_deck*4
        c = N - self.dCount
        c[10] += N*3
        return c[2:12] / self.next_card

    def sCount(self):
        return self.dCount[10:].sum() - self.dCount[2:7].sum()

    def peep_cpr(self):
        return self.sCount() / self.remaining_cards()


class HandBJ:
    def __init__(self):
        self.reset()

    def natural(self):
        return self.sum == 21 and self.n == 2

    def reset(self):
        self.n = 0     # number of cards
        self.sum = 0   # sum of hand
        self.uace = 0  # number of useable ace

    def set(self, h):
        self.n = h.n
        self.sum = h.sum
        self.uace = h.uace

    def soft17_under(self):
        return self.sum - self.uace < 17

    def busted(self):
        return self.sum > 21

    def hit(self, c):
        self.n += 1
        if c == 11:  # ace card
            self.uace += 1
        self.sum += c
        if self.busted() and self.uace > 0:
            self.uace -= 1
            self.sum -= 10
        if self.busted():  # busted
            self.sum = 22
            return True # busted
        else:
            return False


class CasinoBJ:
    def __init__(self):
        self.nDS = 12  # dealer card space
        self.nPS = 23  # player sum space
        self.nPA = 2   # player with/without ace
        self.nA = 4    # action space - 0: hit / 1: stick / 2:double down / 3: surrender
        self.double_factor = 4
        self.deck = CardDeck(1)
        self.pHand = HandBJ()
        self.dHand = HandBJ()
        self.reset_game()

    def reset_game(self):
        self.pHand.reset()
        self.dHand.reset()
        if self.deck.need_shuffle():  # reshuffle only at the beginning of a game
            self.deck.reshuffle()

    def get_action_space(self):
        return self.nA

    def get_state_space(self):
        S = (self.nDS, self.nPS, self.nPA)
        return S

    def observe(self):  # return state
        state = (self.dHand.sum, self.pHand.sum, self.pHand.uace)
        return state

    def peep_cpr(self):
        return self.deck.peep_cpr()

    def peep(self):
        return self.deck.peep()

    def get_card(self):
        return self.deck.get_next_card_BJ()

    def player_hit(self):
        return self.pHand.hit(self.get_card())

    def dealer_hit(self):
        return self.dHand.hit(self.get_card())

    def dealer_turn(self):
        if self.pHand.busted(): # player busted
            return -1    # actually it never happens

        while self.dHand.soft17_under():
            self.dealer_hit()

        if self.pHand.natural() and not self.dHand.natural():
            return 1.2
        if self.dHand.busted():  # dealer busted
            return 1
        if self.dHand.sum < self.pHand.sum :
            return 1
        elif self.dHand.sum == self.pHand.sum :
            # blackjack(10+A) beats other 21
            if self.dHand.natural() and not self.pHand.natural():
                return -1
            reward = 0
        else:
            reward = -1
        return reward

    def start_game(self):
        self.reset_game()
        self.dealer_hit()        #initial card for dealer
        #initial card for player
        self.player_hit()
        self.player_hit()
        # while self.pHand.sum <= 8:
        #     self.player_hit()

    def step(self, action):
        reward = 0
        done = False
        if action == 0 : # hit
            busted = self.player_hit()
            if busted:
                reward = -1
                done = True
        elif action == 1:  # stay
            reward = self.dealer_turn()
            done = True
        elif action == 2: #double down
            busted = self.player_hit()
            if busted:
                reward = -self.double_factor
            else:
                reward = self.dealer_turn() * self.double_factor
            done = True
        else: #action == 3:  #surrender
            reward = -0.5
            done = True

        return self.observe(), reward, done
