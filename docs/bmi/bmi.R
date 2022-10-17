## HEADER ####
## What: bmi script
## Who: Ed
## When: last edited 2021.09.27

## CONTENTS ####
## 00 Setup
## 01 Graphs
## 02 Analysis


## 00 Setup ####

setwd(r'(D:\Dropbox\WORK\__Harper Adams stuff\HARPER courses\_c7081 Statistical Analysis for Data Science\_2021\_labs\bmi)')

# Hard way
# m <- read.delim("bmi_m.txt")
# f <- read.delim("bmi_f.txt")
# bmi <- rbind(m, f)
# bmi$gender <- c(rep("m", nrow(m)), rep("f", nrow(f)))

# Easy way
library(openxlsx)

bmi <- read.xlsx("bmi_male_and_female.xlsx")

## 01 Graphs ####

table(bmi$gender)

par(mfrow=c(2,1))
hist(bmi$bmi[which(bmi$gender == 'male')], 
     col = 'navy')
hist(bmi$bmi[which(bmi$gender == 'female')], 
     col = 'HotPink')
par(mfrow=c(1,1))



## 02 Analysis ####
cor(bmi$bmi, bmi$steps)

lm0 <- lm(bmi ~ steps + gender, data = bmi)


anova(lm0)

names(summary(lm0))
round(summary(lm0)$r.squared, 2)

library(visreg)
visreg(lm0, 'gender')
visreg(lm0, 'steps')

plot(lm0, 1)



# plot it
plot(bmi ~ steps, data = bmi, 
     pch = 16, cex = .7, 
     col = c(rep("HotPink", 921), 
             rep("navy", 865)))

