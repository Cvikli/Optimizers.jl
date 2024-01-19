# Optimizers.jl
Optimizers for Neural networks

## Disclaimer
I tested some flux optimizers too, so their implementation is duplicated and copied for modification.
I was researching for a more adaptive optimizer... so we shouldn't set learning rate... it should figure it out basically. The problem I faced during finding the best algorithm is that batch causes diverging loss, vanishing learning rate, or the learning rate is increased too much and we go NaN, also I cannot recover from infinite values. Everything is somewhat solved, but not perfectly. There must be a solution, so I think later on I will finish with an ultimate optimizer that perform better in every sense (faster converging, higher maximum value and no hyperparameter as it controlled algoritmically based on learning feedback like humans would do.)

