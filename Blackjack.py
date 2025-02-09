import random

class Deck:
    def __init__(self):
        self.draw_pile = []
        self.discard_pile = []
        
    def create(self, amount=1):
        self.draw_pile = []
        a_deck = ['A'] + ['T']*4 + [str(num) for num in range(2,10)]
        self.draw_pile.extend((a_deck*4) * amount)
        random.shuffle(self.draw_pile)

    def deal(self, shuffle=False):
        if len(self.draw_pile) < 1 or shuffle:
            self.draw_pile += self.discard_pile
            self.discard_pile = []
            random.shuffle(self.draw_pile)
            print('- Deck Shufflled')
        return self.draw_pile.pop()
    
    def discard(self, hand):
        self.discard_pile.extend(hand)


class Hand:
    def __init__(self):
        self.hand = []
        self.points = {0}
        
    def hit(self, card):
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
        self.hand = []
        self.points = [0]
    
    def display(self):
        print(f"Hand: {self.hand} | Points: {self.points}")
    

class Score:
    def  __init__(self):
        self.score = [0, 0, 0] # [Wins, Losses, Ties]
    
    def show(self):
        print('Current Stat: Win-%i Lose-%i Tie-%i'
              %(self.score[0], self.score[1], self.score[2]))

    def update(self, idx):
        self.score[idx] += 1
    
    def clear(self):
        self.score = [0, 0, 0]

def is_win(player_points, dealer_points):
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

def main():
    deck = Deck()
    deck.create()
    score = Score()
    player_hand = Hand()
    dealer_hand = Hand()
    
    while True:
        print("\nNew Round!")
        player_hand.clear()
        dealer_hand.clear()
        
        # Initial deal
        player_hand.hit(deck.deal())
        player_hand.hit(deck.deal())
        dealer_hand.hit(deck.deal())
        dealer_hand.hit(deck.deal())
        
        print("Dealer's hand: [Hidden,", dealer_hand.hand[1], "]")
        player_hand.display()
        
        # Player's turn
        while True:
            choice = input("Do you want to (h)it or (s)tand? ").lower()
            if choice == 'h':
                player_hand.hit(deck.deal())
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
                dealer_hand.hit(deck.deal())
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
        score.show()
        
        deck.discard(player_hand.hand)
        deck.discard(dealer_hand.hand)

main()