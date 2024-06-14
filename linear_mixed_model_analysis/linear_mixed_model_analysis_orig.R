# Load packages

rm(list=ls())
cat("\014") 

library(lmerTest)
library(ggplot2)
library(plyr)
library(fitdistrplus)
library(texreg)
library(gridExtra)
library(tidyverse)
library(broom)
library(tidyr)

# Working directory has to contain BolusLoad_nov.csv
#setwd("~/GitHub/Multimodal/linear_mixed_model_analysis/")
setwd("C:/Users/ge84yux/Documents/GitHub/Multimodal-Cleaned/linear_mixed_model_analysis" )
mydata <- read.table("BolusLoad_nov1.csv", header=TRUE, 
                     sep=",",colClasses=c(rep('numeric', 7), 'factor','numeric', rep('factor', 2),'numeric'))

## create new feature for nested ids
#mydata <- na.omit(within(mydata, sample <- factor(id:id2)))
mydata <- (within(mydata, sample <- factor(id:id2)))
mydata <- within(mydata, eventrate <- (1/(IEI/60))) #frame rate was 7.7/sec so divide IEI first by this then convert to min
## rename column and remove extreme samples
colnames(mydata)[8] <- "region"
mydata$region <-revalue(mydata$region, c("1"="V1", "2"="S1" , "3"="AL" , "4"="RL"))
#mydata$rates[mydata$rates >= 1] = 0.99
#mydata <- mydata[which(mydata$rates >= 0.08) , ]
mydata <- mydata[which(mydata$region != "AL") , ]
mydata$age <- mydata$age-8

missing1 <- is.na(mydata$amps)
mydata2 <- subset(mydata, subset = !missing1)

## Fit amplitude as function of age*region and with animal and sample ID as hidden factors
mixed.lmer1 <- lmer(amps ~ age*region +  (1 |id) + (1 |sample) , data = mydata2 )
## check normality of sample
qqnorm(residuals(mixed.lmer1))
qqline(residuals(mixed.lmer1) , col = 2,lwd=2,lty=2)
s1 <- summary(mixed.lmer1)
anova(mixed.lmer1)
print(s1)
confint(mixed.lmer1)

##plot means and look to see if there an exponential fits the data 
meansdata1 = aggregate(mydata$amps, list(mydata$age, mydata$region, mydata$id), mean)
meansdata1 <- setNames(meansdata1,c("age", "region", "id", "amps")) 
ggplot(meansdata1, aes(x = age, y = amps, color = region)) +
  geom_point() +
  geom_smooth(method = "nls", formula = y ~ a * exp(-b * x), 
              se = FALSE, 
              method.args = list(start = list(a = 100, b = 0.1)), 
              aes(group = region)) +
  scale_color_manual(values = c("red", "blue", "green")) +
  labs(x = "Age", y = "Amplitude") +
  theme_minimal()


missing2 <- is.na(mydata$durations)
mydata2 <- subset(mydata, subset = !missing2)

## Fit durations as function of age*region and with animal and sample ID as hidden factors
mixed.lmer2 <- lmer(durations ~ age*region +  (1 |id) + (1 |sample) , data = mydata2 )
qqnorm(residuals(mixed.lmer2))
qqline(residuals(mixed.lmer2) , col = 2,lwd=2,lty=2)
s2 <- summary(mixed.lmer2)
print(s2)
confint(mixed.lmer2)

meansdata2 = aggregate(mydata$durations, list(mydata$age, mydata$region, mydata$id), mean)
meansdata2 <- setNames(meansdata2,c("age", "region", "id", "durations")) 
ggplot(meansdata2, aes(x = age, y = durations, color = region)) +
  geom_point() +
  geom_smooth(method = "nls", formula = y ~ a * exp(-b * x), 
              se = FALSE, 
              method.args = list(start = list(a = 100, b = 0.1)), 
              aes(group = region)) +
  scale_color_manual(values = c("red", "blue", "green")) +
  labs(x = "Age", y = "Duration") +
  theme_minimal()

missing3 <- is.na(mydata$rates)
mydata2 <- subset(mydata, subset = !missing3)

mixed.lmer3 <- lmer(rates ~ age*region +  (1 |id) +  (1 |sample), data = mydata2 )
qqnorm(residuals(mixed.lmer3))
qqline(residuals(mixed.lmer3) , col = 2,lwd=2,lty=2)
s3 <- summary(mixed.lmer3)
print(s3)
confint(mixed.lmer3)

meansdata3 = aggregate(mydata$rates, list(mydata$age, mydata$region, mydata$id), mean)
meansdata3 <- setNames(meansdata3,c("age", "region", "id", "rates")) 
meansdata3 <- (within(meansdata3, logitrate <- log(rates/(1-rates))))
ggplot(data = meansdata3, aes(x=age, y=rates, group=region, col=region)) + geom_point() + geom_smooth(method="lm", se=FALSE)#geom_smooth(method ="lm")
ggplot(meansdata3, aes(x = age, y = rates, color = region)) +
  geom_point() +
  geom_smooth(method = "nls", formula = y ~ a * exp(-b * x), 
              se = FALSE, 
              method.args = list(start = list(a = 100, b = 0.1)), 
              aes(group = region)) +
  scale_color_manual(values = c("red", "blue", "green")) +
  labs(x = "Age", y = "Participation rate") +
  theme_minimal()

missing4 <- is.na(mydata$eventrate)
mydata2 <- subset(mydata, subset = !missing4)

## Fit eventrate as function of age*region and with animal and sample ID as hidden factors
mixed.lmer4 <- lmer(eventrate ~ age*region +  (1 |id) +  (1 |sample) , data = mydata2 )
qqnorm(residuals(mixed.lmer4))
qqline(residuals(mixed.lmer4) , col = 2,lwd=2,lty=2)
s4 <- summary(mixed.lmer4)
print(s4)
confint(mixed.lmer4)

meansdata4 = aggregate(mydata2$eventrate, list(mydata2$age, mydata2$region, mydata2$id), mean)
meansdata4 <- setNames(meansdata4,c("age", "region", "id", "eventrate")) 
ggplot(data = meansdata4, aes(x=age, y=eventrate, group=region, col=region)) + geom_point() + geom_smooth(method="lm", se=FALSE)#, formula= (y ~ exp(x)), se=FALSE)#geom_smooth(method ="lm")
ggplot(meansdata4, aes(x = age, y = eventrate, color = region)) +
  geom_point() +
  geom_smooth(method = "nls", formula = y ~ a * exp(-b * x), 
              se = FALSE, 
              method.args = list(start = list(a = 100, b = 0.1)), 
              aes(group = region)) +
  scale_color_manual(values = c("red", "blue", "green")) +
  labs(x = "Age", y = "Event Rate") +
  theme_minimal()

## Plot effect sizes to file
plotreg(list(mixed.lmer1,mixed.lmer2,mixed.lmer3,mixed.lmer4) , file = "BolusLoadnoAL.pdf" ,
        lwd.vbars = 0 , custom.coef.names = c("Intercept" , "Age", "S1" , "RL" , "S1 : Age" , "RL : Age"), 
        custom.model.names = c("amps","durations","rates","eventrate"),
        override.se = list(s1$coefficients[ , 2] , s2$coefficients[ , 2] , s3$coefficients[ , 2] , s4$coefficients[ , 2]),
        override.pval = list(s1$coefficients[ , 5] , s2$coefficients[ , 5] , s3$coefficients[ , 5] , s4$coefficients[ , 5]))
## Output table with effects
screenreg(list(mixed.lmer1,mixed.lmer2,mixed.lmer3,mixed.lmer4), digits = 3, leading.zero = TRUE , dcolumn = FALSE , sideways = TRUE ,
          override.pvalues = list(s1$coefficients[ , 5] , s2$coefficients[ , 5] , s3$coefficients[ , 5] , s4$coefficients[ , 5] ),
          override.se = list(s1$coefficients[ , 2] , s2$coefficients[ , 2] , s3$coefficients[ , 2] , s4$coefficients[ , 2] ),
          custom.model.names = c("amps","durations","rates","eventrate"))
