import random

class Deck:
    def __init__(self):
        self.draw_pile = []
        self.discard_pile = []
        
    def create(self, amount=1):
        # Create a new deck and shuffle it
        # amount: the number of decks

        self.draw_pile = []
        a_deck = ['A'] + ['T']*4 + [str(num) for num in range(2,10)]
        self.draw_pile.extend((a_deck*4) * amount)
        random.shuffle(self.draw_pile)

    def deal(self, shuffle=False):
        # Draw a card from the draw pile
        # shuffle: add the discard pile to the draw pile after each deal
        
        if len(self.draw_pile) < 1 or shuffle:
            self.draw_pile += self.discard_pile
            self.discard_pile = []
            random.shuffle(self.draw_pile)
            print('- Deck Shufflled')
        return self.draw_pile.pop()
    
    def discard(self, hand):
        # Put the hand of cards to the discard pile
        # hand: a list of cards
        self.discard_pile.extend(hand)


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

    def clear(self):
        # Clear the hand and points
        self.hand = []
        self.points = [0]
    
    def display(self):
        # Display the hand and points
        print(f"Hand: {self.hand} | Points: {self.points}")


class Score:
    def  __init__(self):
        self.score = [0, 0, 0] # [Wins, Losses, Ties]
    
    def display(self):
        # Diplay the scores
        print('Current Stat: Win-%i Lose-%i Tie-%i'
              %(self.score[0], self.score[1], self.score[2]))

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
        return 1  # Loss
    elif dealer_best > 21 or player_best > dealer_best:
        return 0  # Win
    elif player_best < dealer_best:
        return 1  # Loss
    else:
        return 2  # Tie  

def main(shuffle=False):
    deck = Deck()
    deck.create()
    score = Score()
    player_hand = Hand()
    dealer_hand = Hand()
    
    # Initial deal
    player_hand.hit(deck.deal(shuffle))
    player_hand.hit(deck.deal(shuffle))
    dealer_hand.hit(deck.deal(shuffle))
    dealer_hand.hit(deck.deal(shuffle))
    
    dealer_hand_revealed = dealer_hand.hand[1]
    player_hand.hand
    
    while True:
        print("\nNew Round!")
        player_hand.clear()
        dealer_hand.clear()
        
        # Initial deal
        player_hand.hit(deck.deal(shuffle))
        player_hand.hit(deck.deal(shuffle))
        dealer_hand.hit(deck.deal(shuffle))
        dealer_hand.hit(deck.deal(shuffle))
        
        print("Dealer's hand: [Hidden,", dealer_hand.hand[1], "]")
        player_hand.display()
        
        # Player's turn
        while True:
            choice = input("Do you want to (h)it or (s)tand? ").lower()
            if choice == 'h':
                player_hand.hit(deck.deal(shuffle))
                player_hand.display()
                if max(player_hand.points) > 21:
                    print("Bust! You lose.")
                    break
            elif choice == 's':
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
        if result == 0:
            print("You win!")
        elif result == 1:
            print("Dealer wins!")
        else:
            print("It's a tie!")
        
        score.update(result)
        score.display()
        
        deck.discard(player_hand.hand)
        deck.discard(dealer_hand.hand)

if __name__ == "__main__":
    main()