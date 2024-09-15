import matplotlib.pyplot as plt
from IPython import display


plt.ion() # enable interactive mode

def plot(scores, mean_scores): # So that we can plot the scores and the mean scores and see how the agent is doing
  display.clear_output(wait=True) 
  display.display(plt.gcf()) # get current figure
  plt.clf() # clear current figure

  plt.title('Training in progress...')
  plt.xlabel('Number of Games')
  plt.ylabel('Score')
  plt.plot(scores)
  plt.plot(mean_scores)
  plt.ylim(ymin=0)
  plt.text(len(scores)-1, scores[-1], str(scores[-1])) # Display the score at the end of the line
  plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1])) # Display the mean score at the end of the line
  plt.show(block=False)
  plt.pause(.1)
