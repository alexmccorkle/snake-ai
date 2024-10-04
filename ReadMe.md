# Snake AI

## Following a [tutorial](https://www.youtube.com/watch?v=L8ypSXwyBds) to make Snake + Train an AI to play it

Weekend shenanigans

---

Post-Video Additions:

- Added Save/Load functionality
  - AI continues from where you left off. Record and total score is also saved/loaded.
  - Update model to prevent the spinning bad habit.
  - Added negative rewards for trapping itself.
    - But this needs to be optimized a bit more for 'wider' decisions that get itself trapped.

## TODO

- Update trapped method to see if it'll less often get stuck in a big loop.
- See if I can optimize the model any further to see if it can average a higher score
  - Currently slows down around 35 points or so.

---
