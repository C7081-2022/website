## HEADER ####
## who:
## what:
## when:

## CONTENTS ####
## 00 Setup
## 01 solution

# word problem
# bird 1: x1 + 1 = X2 - 1
# bird 1: x1 - x2 = -2

# bird 2: x1 - 1 = 2*(X2 + 1)
# bird 2: x1-2x2 = 3

A <- matrix(c(1, 1, -1, -2), ncol = 2)
b <- c(-2, 3)
(x <- solve(A) %*% b)



# bird 1: x1 = -2 + x2
# bird 2: x1 = 2 + 2 * x2
plot(x=NULL, y=NULL,
     xlim = c(-8,3), ylim = c(-8,3),
     pch = 16, 
     xlab=expression('x'[1]), 
     ylab=expression('x'[2]))
abline(h=c(-8:3),v=c(-8:3),
       lty=2, col='green3')
abline(h=0, v=0, lwd=2)

abline(a=-2, b=-1, col="red", lwd=2)
abline(a=3, b=2, col="red", lwd=2)
