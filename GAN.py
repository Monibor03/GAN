# GAN to produce the distribution of age for eICU patients.

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Load dataset
#Data Directory: https://github.com/JeffersonLab/jlab_datascience_data/blob/main/eICU_age.npy
ages = np.load("C:\\Users\\monibor\\Desktop\\Github code\\GAN/eICU_age.npy")
ages = ages.reshape(-1, 1).astype(np.float32)
print(ages.shape)
# Min-Max Normalization
min_age, max_age = ages.min(), ages.max()
ages = (ages - min_age) / (max_age - min_age)

# Model Parameter
latent_dim = 64
epochs = 5000
batch_size = 64

# Generator
def build_generator():
    model = models.Sequential([
        layers.Input(shape=(latent_dim,)),
        layers.Dense(64),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(128),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(64),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(1, activation='linear')
    ])
    return model

# Discriminator
def build_discriminator():
    model = models.Sequential([
        layers.Input(shape=(1,)),
        layers.Dense(64),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(128),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(64),
        layers.LeakyReLU(alpha=0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Models
generator = build_generator()
discriminator = build_discriminator()

# Compile discriminator
discriminator.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

# Compile Gen and Dis together
discriminator.trainable = False
z = layers.Input(shape=(latent_dim,))
fake_age = generator(z)
validity = discriminator(fake_age)
combined = models.Model(z, validity)
combined.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
                 loss='binary_crossentropy')

# Training process
valid = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

for epoch in range(epochs):
    # Train Discriminator ()
    idx = np.random.randint(0, ages.shape[0], batch_size)
    real_ages = ages[idx]

    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    gen_ages = generator.predict(noise, verbose=0)

    d_loss_real = discriminator.train_on_batch(real_ages, valid)
    d_loss_fake = discriminator.train_on_batch(gen_ages, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train Generator
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    g_loss = combined.train_on_batch(noise, valid)

    if epoch % 500 == 0:
        print(f"{epoch} [D loss: {d_loss[0]:.4f}, acc: {100*d_loss[1]:.2f}%] [G loss: {g_loss:.4f}]")

# Generate fake samples using trained generator
noise = np.random.normal(0, 1, (2520, latent_dim))
gen_ages = generator.predict(noise, verbose=0)

# Denormalize to real age range
real_ages_denorm = ages *(max_age - min_age) + min_age
gen_ages_denorm = gen_ages * (max_age - min_age) + min_age

min_age = min(real_ages_denorm.min(), gen_ages_denorm.min())
max_age = max(real_ages_denorm.max(), gen_ages_denorm.max())
bins = np.linspace(min_age, max_age, 31)  # 30 bins means 31 edges


# Plot distribution
plt.figure(figsize=(8,5))
plt.hist(real_ages_denorm, bins=bins, alpha=0.6, label='True distribution')
plt.hist(gen_ages_denorm, bins=bins, alpha=0.6, label='Generated distribution')
plt.legend()
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("True vs Generated Age Distribution")
plt.show()