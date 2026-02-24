# Calculus Review 
# Assignment 3
# Completed questions 1 to 4 in google document 

# 5.R Code & Gradient Logic
# Initial data and parameters
x <- c(0,0,0,0,0,1,1,1,1,1)
y <- c(0,1,0,0,0,0,1,1,1,0)
b0 <- 0
b1 <- 0

# Calculating Gradients
pi <- 1 / (1 + exp(-(b0 + b1 * x)))
dJ0 <- sum(pi - y)
dJ1 <- sum(x * (pi - y))

cat("Initial Gradient for b0 (dJ0):", dJ0, "\n")
# Answer: 1

cat("Initial Gradient for b1 (dJ1):", dJ1, "\n")
# Answer: -0.5

# 6. Gradient Descent Implementation
lambda <- 0.01

# Loop for 5000 iterations to ensure high precision convergence
for (i in 1:5000) {
  # Recomputing probabilities with updated parameters
  pi_current <- 1 / (1 + exp(-(b0 + b1 * x)))
  
  # Recomputing gradients
  dJ0 <- sum(pi_current - y)
  dJ1 <- sum(x * (pi_current - y))
  
  # Updating parameters
  b0 <- b0 - lambda * dJ0
  b1 <- b1 - lambda * dJ1
}

print(paste("Converged b0:", round(b0, 4)))
# Answer: Converged b0: -1.3863

print(paste("Converged b1:", round(b1, 4)))
# Answer: Converged b1: 1.7918

# 7. Convert estimates to probabilities
# Formula: P(Y=1|x) = 1 / (1 + exp(-(b0 + b1*x)))

prob_x0 <- 1 / (1 + exp(-(b0 + b1 * 0)))
prob_x1 <- 1 / (1 + exp(-(b0 + b1 * 1)))

cat("P(Y = 1 | x = 0) =", round(prob_x0, 2), "\n")
# Answer: P(Y = 1 | x = 0) = 0.2

cat("P(Y = 1 | x = 1) =", round(prob_x1, 2), "\n")
# Answer: P(Y = 1 | x = 1) = 0.6 
