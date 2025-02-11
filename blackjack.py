import random

class Deck:
    def __init__(self, amount=2):
        self.amount = amount
        self.draw_pile = []
        self.discard_pile = []
        
        self.create()
        
    def create(self):
        # Create a new deck and shuffle it
        a_deck = ['A']*4 + ['T']*16 + [str(num) for num in range(2,10)]*4
        self.draw_pile = a_deck * self.amount
        random.shuffle(self.draw_pile)

    def deal(self, shuffle=False):
        # Draw a card from the draw pile
        # shuffle: add the discard pile to the draw pile after each deal
        if len(self.draw_pile) < 1 or shuffle:
            self.draw_pile += self.discard_pile
            self.discard_pile = []
            random.shuffle(self.draw_pile)
        return self.draw_pile.pop()
    
    def discard(self, hand):
        # Put the hand of cards to the discard pile
        self.discard_pile.extend(hand)


class CardCountingDeck(Deck):
    def __init__(self, amount=2):
        self.amount = amount
        self.draw_pile = []
        self.discard_pile = []
        
        self.card_counts = [4*amount for i in range(1, 10)] + [16*amount]
        self.card_sum = 52 * amount
        
        self.create()
    
    def deal(self, shuffle=False):
        # Draw a card from the draw pile
        if len(self.draw_pile) < 1 or shuffle:
            self.draw_pile += self.discard_pile
            self.discard_pile = []
            random.shuffle(self.draw_pile)
            
            self.card_counts = [4*self.amount for i in range(1, 10)] + [16*self.amount]
            self.card_sum = 52 * self.amount
        return self.draw_pile.pop()
    
    def update_counts(self, hand):
        for card in hand:
            if card == 'A':
                idx = 0
            elif card == 'T':
                idx = 9
            else:
                idx = int(card) - 1
            self.card_counts[idx] -= 1
            self.card_sum -= 1
    
    @property
    def probabilities(self):
        return [i / self.card_sum for i in self.card_counts]


class Hand:
    def __init__(self):
        self.hand = []
        self.points = {0}
        
    def hit(self, card):
        # Put the card to the hand and compute the points
        self.hand.append(card)
        
        if card == 'T':
            total_points = [x + 10 for x in self.points]
        elif card == 'A':
            total_points = [x + 1 for x in self.points] + [x + 11 for x in self.points]
            total_points.sort()
        else:
            total_points = [x + int(card) for x in self.points]
        self.points = {p for p in total_points if p <= 21} or {min(total_points)}
    
    def display(self):
        # Display the hand and points
        print(f"Hand: {self.hand} | Points: {self.points}")


class Score:
    def  __init__(self):
        self.score = [0, 0, 0] # [Ties, Wins, Losses]
    
    def display(self):
        # Diplay the scores
        if self.score[1] == 0:
            win_lose_rate = 0
        else:
            win_lose_rate = round(self.score[1]/(self.score[1] + self.score[2]) * 100)
        print('Stat: Win-%i Lose-%i Tie-%i | Win Rate: %i%%'
              %(self.score[1], self.score[2], self.score[0], win_lose_rate))

    def update(self, idx):
        # Update the scores
        self.score[idx] += 1
    
    def clear(self):
        # Clear the scores
        self.score = [0, 0, 0]


def is_win(player_points, dealer_points):
    """
    Return 0 if player wins, 1 if dealer wins and 2 if it is a tie
    """
    player_best = max(player_points)
    dealer_best = max(dealer_points)
    
    if player_best > 21:
        return -1  # Loss
    elif dealer_best > 21 or player_best > dealer_best:
        return 1  # Win
    elif player_best < dealer_best:
        return -1  # Loss
    else:
        return 0  # Tie  

def main(shuffle=False):
    deck = Deck()
    score = Score()
    
    running = True
    while running:
        print("\nNew Round!")
        player_hand = Hand()
        dealer_hand = Hand()
        
        # Initial deal
        player_hand.hit(deck.deal(shuffle))
        player_hand.hit(deck.deal(shuffle))
        dealer_hand.hit(deck.deal(shuffle))
        dealer_hand.hit(deck.deal(shuffle))
        
        print("Dealer's hand: [Hidden, " + dealer_hand.hand[1] + "]")
        player_hand.display()
        
        # Player's turn
        while True:
            choice = input("Do you want to (h)it or (s)tand?").lower()
            if choice == 'h':
                player_hand.hit(deck.deal(shuffle))
                player_hand.display()
                if max(player_hand.points) > 21:
                    print("Bust! You lose.")
                    break
                if max(player_hand.points) == 21:
                    print("Blackjack!")
                    break                    
            elif choice == 's':
                break
            elif choice == 'q':
                running = False
                break
            else:
                print("Invalid input. Please choose 'h' or 's'.")
        
        # Dealer's turn
        if max(player_hand.points) <= 21:
            while max(dealer_hand.points) < max(player_hand.points):
                dealer_hand.hit(deck.deal(shuffle))
        print("Dealer's final hand:")
        dealer_hand.display()
        
        # Determine result
        result = is_win(player_hand.points, dealer_hand.points)
        if result == 1:
            print("You win!")
        elif result == -1:
            print("Dealer wins!")
        else:
            print("It's a tie!")
        
        score.update(result)
        score.display()
        
        deck.discard(player_hand.hand)
        deck.discard(dealer_hand.hand)

if __name__ == "__main__":
    main()
