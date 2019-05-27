# Latent Space Exploration of Generative Adversarial Networks
This project was conducted as a master thesis by Erik Sandström during the spring of 2019 at Lund University.

This README-file contains extra material for Section 7.2.1 and 7.2.2 not covered in the thesis report. 

Below, the random walk interpolations are presented for each model when restricting the latent space to only vary the dimensions {1-8,9-16,17-32,33-64,65-128,129-256} at a time. 
# clgGAN
## Restricted to dimensions: 1-8
![](gifs/clgGAN/1-8_5pts_100_50.gif)

## Restricted to dimensions: 9-16
![](gifs/clgGAN/9-16_5pts_100_50.gif)

## Restricted to dimensions: 17-32
![](gifs/clgGAN/17-32_5pts_100_50.gif)

## Restricted to dimensions: 33-64
![](gifs/clgGAN/33-64_5pts_100_50.gif)

## Restricted to dimensions: 65-128
![](gifs/clgGAN/65-128_5pts_100_50.gif)

## Restricted to dimensions: 129-256
![](gifs/clgGAN/129-256_5pts_100_50.gif)

# cgGAN

# bgGAN

# lgGAN

# bGAN

Below, the random walk interpolations along the coordinate axis for the clgGAN are presented when restricting the latent space to only along the coordinate axis {1,2,5,90,195,250} at a time. 

# Setup
Place a directory called "celebA_dataset" where the model directory, e.g., "clgGAN" is located. In the "celebA_dataset" directory, place another folder called "celebA" containing the images belonging to the dataset.
