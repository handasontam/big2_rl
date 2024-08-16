# Big 2 Self-Play Reinforcement Learning AI


- This repo is largely based on https://github.com/henrycharlesworth/big2_PPOalgorithm
- Scoring rules is set according to https://big2.lihkg.com/ (see the following).
- Game rules changed to Hong Kong style:
  - Don't allow JQKA2, QKA23, KA234
  - Allow A2345 and 23456
  - Don't allow playing Four-of-a-kind without a single card.
  - Allow playing King Kong (i.e. Four-of-a-kind + One card)
  - Don't allow playing two pairs.
- Implemented the following RL algorithms with additional ways to handle dynamic action space (to be documented...):
  - Neural Replicator Dynamics
  - PPO
- It is super-human level already according to my own experience.

# Play with the AI with the trained model
```
conda env create -f environment.yml
conda activate big2torch
python generateGUI_torch.py
```

# Game Scoring Rules

## 完GAME時，以剩下手牌計分。 When the game ends, score based on remaining cards in hand.

## 基本分數 (Basic Scoring):

- 少於八張，每張一分；
  Less than 8 cards, 1 point per card;

- ≥八張即雙炒(剩下手牌數 x 2)；
  ≥8 cards is "double penalty" (remaining cards x 2);

- ≥十張即三炒(剩下手牌數 x 3)；
  ≥10 cards is "triple penalty" (remaining cards x 3);

- 十三張即四炒(剩下手牌數 x 4)。
  13 cards is "quadruple penalty" (remaining cards x 4).

**P.S.** 第一個出牌嘅玩家，手牌數≥七張即雙妙。

**P.S.** For the first player to play a card, ≥7 cards is "double wonderful".

## 最終得分 (Final Score):

- A的得分=(B的牌分-A的牌分)+(C的牌分-A的牌分)+(D的牌分-A的牌分)
- A's score = (B's card points - A's card points) + (C's card points - A's card points) + (D's card points - A's card points)

- B的得分=(A的牌分-B的牌分)+(C的牌分-B的牌分)+(D的牌分-B的牌分)
- B's score = (A's card points - B's card points) + (C's card points - B's card points) + (D's card points - B's card points)

- C的得分=(A的牌分-C的牌分)+(B的牌分-C的牌分)+(D的牌分-C的牌分)
- C's score = (A's card points - C's card points) + (B's card points - C's card points) + (D's card points - C's card points)

- D的得分=(A的牌分-D的牌分)+(B的牌分-D的牌分)+(C的牌分-D的牌分)
- D's score = (A's card points - D's card points) + (B's card points - D's card points) + (C's card points - D's card points)

**P.S.** 第一名額外 +10 分獎勵
**P.S.** The first place gets an additional +10 points bonus


# TODO
- Code refactoring and improve the game simulation efficiency.
- Implement a web UI.
- Implement the DeepNash model: https://arxiv.org/abs/2206.15378
- Document this new idea of handling dynamic action space.
- Containerized the application.


