import json
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import random
from scipy.stats import entropy
import matplotlib.pyplot as plt

# Load the data
with open('intents.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Extract tags and their corresponding patterns
tags = [intent['tag'] for intent in data['intents']]
patterns = [' '.join(intent['patterns']) for intent in data['intents']]

# Load a pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for the patterns
embeddings = model.encode(patterns)

# Perform hierarchical clustering
n_clusters = 10  # You can adjust this number
clustering = AgglomerativeClustering(n_clusters=n_clusters)
clustering.labels_ = clustering.fit_predict(embeddings)

# Create a mapping from original tags to cluster labels
tag_mapping = dict(zip(tags, clustering.labels_))


# Function to get the most common word in a list of tags
def get_common_word(tags):
    words = [word for tag in tags for word in tag.split('-')]
    return max(set(words), key=words.count)


# Create descriptive names for each cluster
cluster_names = {}
for cluster in range(n_clusters):
    cluster_tags = [tag for tag, label in tag_mapping.items() if label == cluster]
    cluster_names[cluster] = f"{get_common_word(cluster_tags)}-group"

# Update the tag mapping with descriptive names
tag_mapping = {tag: cluster_names[label] for tag, label in tag_mapping.items()}

# Create a new DataFrame with the grouped tags
rows = []
for intent in data['intents']:
    tag = intent['tag']
    patterns = intent['patterns']
    responses = intent['responses']
    grouped_tag = tag_mapping[tag]

    if len(responses) == 1:
        response = responses[0]
        for pattern in patterns:
            rows.append({'original_tag': tag, 'grouped_tag': grouped_tag, 'pattern': pattern, 'response': response})
    elif len(patterns) > len(responses):
        extended_responses = random.choices(responses, k=len(patterns))
        for pattern, response in zip(patterns, extended_responses):
            rows.append({'original_tag': tag, 'grouped_tag': grouped_tag, 'pattern': pattern, 'response': response})
    else:
        for pattern, response in zip(patterns, responses):
            rows.append({'original_tag': tag, 'grouped_tag': grouped_tag, 'pattern': pattern, 'response': response})

df = pd.DataFrame(rows)

# Print the distribution of grouped tags
print("Distribution of grouped tags:")
print(df['grouped_tag'].value_counts())

# Save the updated DataFrame to a CSV file
df.to_csv('grouped_intents.csv', index=False, encoding='utf-8')

# Print some sample rows to verify the grouping
print("\nSample rows from the grouped dataset:")
print(df.sample(5))

# Save the tag mapping for future reference
with open('tag_mapping.json', 'w', encoding='utf-8') as f:
    json.dump(tag_mapping, f, ensure_ascii=False, indent=2)

print("\nGrouping complete. Results saved to 'grouped_intents.csv' and 'tag_mapping.json'.")

# Analyze the new distribution
print("\nNew tag distribution:")
new_distribution = df['grouped_tag'].value_counts()
print(new_distribution)

print("\nDescriptive statistics of the new distribution:")
print(new_distribution.describe())

# Calculate and print the entropy of the new distribution
new_probabilities = new_distribution / new_distribution.sum()
new_entropy = entropy(new_probabilities)
print(f"\nEntropy of the new distribution: {new_entropy:.4f}")

# Compare with the original distribution
original_distribution = df['original_tag'].value_counts()
original_probabilities = original_distribution / original_distribution.sum()
original_entropy = entropy(original_probabilities)
print(f"Entropy of the original distribution: {original_entropy:.4f}")

print(f"\nNumber of original tags: {len(original_distribution)}")
print(f"Number of grouped tags: {len(new_distribution)}")

# Visualize the new distribution
plt.figure(figsize=(12, 6))
new_distribution.plot(kind='bar')
plt.title('Distribution of Grouped Tags')
plt.xlabel('Grouped Tags')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('grouped_tags_distribution.png')
print("\nDistribution plot saved as 'grouped_tags_distribution.png'")