import tensorflow_datasets as tfds

# Load the dataset
dataset, info = tfds.load('cats_vs_dogs', with_info=True, as_supervised=True)

# Split into training and testing sets
train_dataset = dataset['train']
test_dataset = dataset['test']

# Print dataset information
print(info)