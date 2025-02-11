import torch
import torch.nn as nn
import torch.optim as optim
import random
import tqdm
from collections import deque
from blackjack import *


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),            
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.model(x)


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000)
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32

        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(next_state_tensor)).item()

            target_f = self.model(state_tensor).detach().clone()
            target_f[0][action] = target

            # Perform gradient descent
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(state_tensor), target_f)
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, path="blackjack_dqn.pth"):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load_model(self, path="blackjack_dqn.pth"):
        import os
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
            self.model.eval()
            print(f"Model loaded from {path}")
        else:
            print("Model file not found. Starting with a new model.")


def encode_state(player_hand, dealer_revealed_card, card_counts):
    """
    Encode the game state as a feature vector using the Hand class to compute points.
    """
    # Get player hand points and dealer revealed card points
    player_points = list(player_hand.points)
    dealer_hand = Hand()
    dealer_hand.hit(dealer_revealed_card)
    dealer_points = list(dealer_hand.points)

    # Flatten the points and discard values for the state representation
    state = [0]*(2-len(player_points)) + player_points + [0]*(2-len(dealer_points)) + dealer_points + card_counts # 2 + 2 + 10
    
    return state


def train(agent, episodes=2000):
    deck = CardCountingDeck()
    score = Score()
    
    for episode in range(episodes):
        player_hand = Hand()
        dealer_hand = Hand()

        # Initial deal
        player_hand.hit(deck.deal())
        player_hand.hit(deck.deal())
        dealer_hand.hit(deck.deal())
        dealer_hand.hit(deck.deal())
        
        dealer_revealed_card = dealer_hand.hand[0]
        deck.update_counts(player_hand.hand + [dealer_revealed_card])

        state = encode_state(player_hand, dealer_revealed_card, deck.card_counts)

        done = False
        while not done:
            action = agent.act(state)

            # 0 = hit, 1 = stand
            if action == 0:  # Player hits
                deal = deck.deal()
                player_hand.hit(deal)
                deck.update_counts([deal])
                if max(player_hand.points) > 21:
                    reward = -1
                    done = True                  
                else:
                    reward = 0.1
            else:  # Player stands
                # Dealer's turn
                while max(dealer_hand.points) < max(player_hand.points):
                    dealer_hand.hit(deck.deal())

                reward = is_win(player_hand.points, dealer_hand.points)
                done = True

            next_state = encode_state(player_hand, dealer_revealed_card, deck.card_counts)
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                deck.discard(player_hand.hand)      
                deck.discard(dealer_hand.hand)
                
                deck.update_counts(dealer_hand.hand[1:])
                
                score.update(reward)
                if episode % 500 == 0:
                    print(f"Episode {episode} ", end='')
                    score.display()
                break

        agent.replay()

    print(f"Episode {episodes} ", end='')
    score.display()    
    agent.save_model()


def test(agent, episodes=10000):
    deck = CardCountingDeck()
    score = Score()
    
    for episode in range(episodes):
        player_hand = Hand()
        dealer_hand = Hand()
    
        # Initial deal
        player_hand.hit(deck.deal())
        player_hand.hit(deck.deal())
        dealer_hand.hit(deck.deal())
        dealer_hand.hit(deck.deal())
    
        dealer_revealed_card = dealer_hand.hand[0]
        deck.update_counts(player_hand.hand + [dealer_revealed_card])

        state = encode_state(player_hand, dealer_revealed_card, deck.card_counts)
        
        while True:
            action = agent.act(state)
            
            done = False
            if action == 0:  # Player hits
                deal = deck.deal()
                player_hand.hit(deal)
                deck.update_counts([deal])
                if max(player_hand.points) > 21:
                    done = True
            else:
                # Dealer's turn
                while max(dealer_hand.points) < max(player_hand.points):
                    dealer_hand.hit(deck.deal())
                done = True
            
            if done:
                deck.discard(player_hand.hand)      
                deck.discard(dealer_hand.hand)
                
                deck.update_counts(dealer_hand.hand[1:])
                
                score.update(is_win(player_hand.points, dealer_hand.points))
                if episode % 2000 == 0:
                    print(f"Episode {episode} ", end='')
                    score.display()
                break
            
            state = encode_state(player_hand, dealer_revealed_card, deck.card_counts)
    print(f"Episode {episodes} ", end='')
    score.display()

if __name__ == "__main__":
    state_size = 14
    action_size = 2  # 2 actions: hit or stand
    
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    agent = DQNAgent(state_size, action_size)
    agent.load_model()    
    train(agent)
    
    print("\nTest:")
    test(agent)
