import pandas as pd
import matplotlib.pyplot as plt

# Sample DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'Score': [85, 90, 88]}
df = pd.DataFrame(data)

# Create a figure and a set of subplots
fig, ax = plt.subplots()

# Plot a bar chart using the 'Score' column
ax.bar(df['Name'], df['Score'], color=['#4CAF50', '#2196F3', '#FFC107'])

# Add titles and labels
ax.set_title('Scores by Name', fontsize=14, fontweight='bold')
ax.set_ylabel('Score', fontsize=12)
ax.set_xlabel('Name', fontsize=12)

# Hide axes for the table area
ax2 = ax.twinx()
ax2.xaxis.set_visible(False) 
ax2.yaxis.set_visible(False)
ax2.set_frame_on(False)

# Create a table from the DataFrame
table = ax2.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='bottom')

# Adjust the layout to make room for the table
plt.subplots_adjust(left=0.2, bottom=0.2)

# Resize the table to make it fit well
table.scale(1, 1.5)
table.auto_set_font_size(False)
table.set_fontsize(10)

# Display the plot
plt.show()
