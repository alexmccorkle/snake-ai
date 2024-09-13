# Snake AI

## Following a [tutorial](https://www.youtube.com/watch?v=L8ypSXwyBds) to make Snake + Train an AI to play it

Weekend shenanigans

---

Will be using PyGame to make Snake (A very simple version of it) and will be using PyTorch for the Model.

Model will take in 11 numbers (states) and return 3 output numbers (probability of going straight, left or right)

Notes (I learn better by writing):

---

Q-Learning:
Q Value = Quality of Action 0. Init Q Value (= init model)

1. Choose action (model.predict(state))
2. Perform action
3. Measure reward
4. Update Q-Value (+ train model)
   Repeat 1-4

---

After this I hope to do the same for other games I know of, but will try to do these without the guides for learning's sake.
