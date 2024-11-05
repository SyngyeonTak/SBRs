import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

# Original attention scores tensor
attention_scores = torch.tensor([0.2371, 0.2698, 0.2624, 0.2698, 0.2371, 0.2624, 0.2698, 0.2400, 0.2470,
                                 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                                 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000])

# Normalize the scores to range [0, 1]
min_score = attention_scores.min()
max_score = attention_scores.max()

# Avoid division by zero if all values are the same
if max_score - min_score != 0:
    normalized_scores = (attention_scores - min_score) / (max_score - min_score)
else:
    normalized_scores = attention_scores

# Convert to numpy array for plotting
normalized_scores_np = normalized_scores.numpy()

# Replace zero values with NaN
normalized_scores_np[normalized_scores_np == 0] = np.nan

normalized_scores_reshaped = normalized_scores_np.reshape(1, -1)

# Create the 1D heat map
plt.figure(figsize=(12, 2))  # Adjust the figure size as needed
plt.imshow(normalized_scores_reshaped, aspect='auto', cmap='viridis')
plt.colorbar(label='Attention Score')
plt.title('1D Attention Scores Heat Map')
plt.yticks([])  # Hide the y-axis ticks
plt.xticks(np.arange(len(normalized_scores_np)), np.arange(1, len(normalized_scores_np)+1))
plt.xlabel('Position')
plt.show()