# Blackjack-Deep-Q-Networks

The win ratio is around 43% while using 2 decks of cards. Not able to split.

- `train.py`: train and test the counting-card model
- `train_shuffle.py`: train and test the non-counting-card model
- `blackjack.py`: a rough version of Blackjack. Able to play manually

## `train.py`
- This DQN counts cards
- The dealer stop hitting only when it has a higher point than the agent

## `train_shuffle.py`
- Shuffle the deck after every game, meaning the card counting is useless
- Make decision based on the card on hand and dealer's revealed card

*You will 100% lose money from Blackjack as you play more and more, no matter how*

## Prerequisites
- Pytorch
- tqdm
- Cuda

## To-Do
- [ ] Write more comments
