# Blackjack-Deep-Q-Networks

The win ratio is around 43% while using 2 decks of cards. Not able to split.

- `train_count.py`: train and test the counting-card model
- `train_shuffle.py`: train and test the non-counting-card model
- `blackjack.py`: a rough version of Blackjack. Able to play manually
- The dealer stop hitting only when it has a higher point than the agent

## `train_count.py`
- This DQN counts cards to compute the probabilities of each number
- Win rate of 43%, need more fine-tuning

## `train_shuffle.py`
- Shuffle the deck after every game, meaning the card counting is useless
- Win rate of 43%

*You will 100% lose money from Blackjack as you play more and more, no matter how*

## Prerequisites
- Pytorch
- tqdm
- Cuda

## To-Do
- [ ] Write more comments
