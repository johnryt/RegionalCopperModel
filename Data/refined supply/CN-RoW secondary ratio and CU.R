# Import Data and require packages
library(plm)
#library(stargazer)
library(lmtest)

# Set path, read data
johnryterwd <- ("C:/Users/ryter/Dropbox (MIT)/Group Research Folder_Olivetti/Displacement/00 Simulation/06 Module integration/Other scenarios/China import ban/Integration/Data/refined supply")
setwd(johnryterwd)
ref_data <- read.csv("Refinery data calculated from WoodMac", header = TRUE, sep = ",", na.strings = c("NA","#DIV/0!"))
cn_primary <- read.csv("Panel data - cn primary refineries.csv", header = TRUE, sep = ",", na.strings = c("NA","#DIV/0!"))
cn_secondary <- read.csv("Panel data - cn secondary refineries.csv", header = TRUE, sep = ",", na.strings = c("NA","#DIV/0!"))
cn_total <- read.csv("Panel data - cn total refineries.csv", header = TRUE, sep = ",", na.strings = c("NA","#DIV/0!"))
rw_primary <- read.csv("Panel data - row primary refineries.csv", header = TRUE, sep = ",", na.strings = c("NA","#DIV/0!"))
rw_secondary <- read.csv("Panel data - row secondary refineries.csv", header = TRUE, sep = ",", na.strings = c("NA","#DIV/0!"))
rw_total <- read.csv("Panel data - row total refineries.csv", header = TRUE, sep = ",", na.strings = c("NA","#DIV/0!"))

# Panel data
p_cn_primary <- pdata.frame(cn_primary,index=c("X","Year"), drop.index=TRUE, row.names=TRUE)
p_cn_secondary <- pdata.frame(cn_secondary,index=c("X","Year"), drop.index=TRUE, row.names=TRUE)
p_cn_total <- pdata.frame(cn_total,index=c("X","Year"), drop.index=TRUE, row.names=TRUE)
p_rw_primary <- pdata.frame(rw_primary,index=c("X","Year"), drop.index=TRUE, row.names=TRUE)
p_rw_secondary <- pdata.frame(rw_secondary,index=c("X","Year"), drop.index=TRUE, row.names=TRUE)
p_rw_total <- pdata.frame(rw_total,index=c("X","Year"), drop.index=TRUE, row.names=TRUE)

####################################################### Random Effects #########################################################

# Model 1.1: Random effects, China primary refineries
re_cn_pri <- plm(log(CU) ~ log(TCRC)+ log(SP2), 
             data=p_cn_primary, model='random', effect='individual')
summary(re_cn_pri, vnpv = vnpvHC)

# Model 1.2: Random effects, RoW primary refineries
re_rw_pri <- plm(log(CU) ~ log(TCRC)+ log(SP2),
            data=p_rw_primary, model='random', effect='individual')#, na.action=na.exclude)
summary(re_rw_pri, vnpv = vnpvHC)

# Model 1.3: Random effects, China secondary refineries
re_cn_sec <- plm(log(CU) ~ log(TCRC) + log(SP2),
                 data=p_cn_secondary, model='random', effect='individual')
summary(re_cn_sec, vnpv = vnpvHC)

# Model 1.4: Random effects, RoW secondary refineries
re_rw_sec <- plm(log(CU) ~ log(TCRC) + log(SP2), 
                 data=p_rw_secondary, model='random', effect='individual')#, na.action=na.exclude)
summary(re_rw_sec, vnpv = vnpvHC)

# Model 1.5: Random effects, China all refineries
re_cn_tot <- plm(log(CU) ~ log(TCRC) + log(SP2), 
                 data=p_cn_total, model='random', effect='individual')
summary(re_cn_tot, vnpv = vnpvHC)

# Model 1.6: Random effects, RoW all refineries
re_rw_tot <- plm(log(CU) ~ log(TCRC), 
                 data=p_rw_total, model='random', effect='individual')#, na.action=na.exclude)
summary(re_rw_tot, vnpv = vnpvHC)

################################################# Fixed Effects ###################################################################

# Model 2.1: Fixed effects, China primary refineries
fe_cn_pri <- plm(log(CU) ~ log(TCRC), #+ log(SP2), 
                 data=p_cn_primary, model='within', effect='individual')
summary(fe_cn_pri, vnpv = vnpvHC)

# Model 2.2: Fixed effects, RoW primary refineries
fe_rw_pri <- plm(log(CU) ~ log(TCRC), #+ log(SP2),
                 data=p_rw_primary, model='within', effect='individual')#, na.action=na.exclude)
summary(fe_rw_pri, vnpv = vnpvHC)

# Model 2.3: Fixed effects, China secondary refineries
fe_cn_sec <- plm(log(CU) ~ log(TCRC), # + log(SP2),
                 data=p_cn_secondary, model='within', effect='individual')
summary(fe_cn_sec, vnpv = vnpvHC)

# Model 2.4: Fixed effects, RoW secondary refineries
fe_rw_sec <- plm(log(CU) ~ log(TCRC), # + log(SP2), 
                 data=p_rw_secondary, model='within', effect='individual')#, na.action=na.exclude)
summary(fe_rw_sec, vnpv = vnpvHC)

# Model 2.5: Fixed effects, China all refineries
fe_cn_tot <- plm(log(CU) ~ log(TCRC), # + log(SP2), 
                 data=p_cn_total, model='within', effect='individual')
summary(fe_cn_tot, vnpv = vnpvHC)

# Model 2.6: Fixed effects, RoW all refineries
fe_rw_tot <- plm(log(CU) ~ log(TCRC), 
                 data=p_rw_total, model='within', effect='individual')#, na.action=na.exclude)
summary(fe_rw_tot, vnpv = vnpvHC)

##################################################### First Difference ###############################################################

# Model 3.1: First difference, China primary refineries
fd_cn_pri <- plm(log(CU) ~ log(TCRC), # + log(SP2), 
                 data=p_cn_primary, model='fd', effect='individual')
summary(fd_cn_pri, vnpv = vnpvHC)

# Model 3.2: First difference, RoW primary refineries
fd_rw_pri <- plm(log(CU) ~ log(TCRC),# + log(SP2),
                 data=p_rw_primary, model='fd', effect='individual')#, na.action=na.exclude)
summary(fd_rw_pri, vnpv = vnpvHC)

# Model 3.3: First difference, China secondary refineries
fd_cn_sec <- plm(log(CU) ~ log(TCRC),# + log(SP2),
                 data=p_cn_secondary, model='fd', effect='individual')
summary(fd_cn_sec, vnpv = vnpvHC)

# Model 3.4: First difference, RoW secondary refineries
fd_rw_sec <- plm(log(CU) ~ log(TCRC),# + log(SP2), 
                 data=p_rw_secondary, model='fd', effect='individual')#, na.action=na.exclude)
summary(fd_rw_sec, vnpv = vnpvHC)

# Model 3.5: First difference, China all refineries
fd_cn_tot <- plm(log(CU) ~ log(TCRC),# + log(SP2), 
                 data=p_cn_total, model='fd', effect='individual')
summary(fd_cn_tot, vnpv = vnpvHC)

# Model 3.6: First difference, RoW all refineries
fd_rw_tot <- plm(log(CU) ~ log(TCRC),# + log(SP2), 
                 data=p_rw_total, model='fd', effect='individual')#, na.action=na.exclude)
summary(fd_rw_tot, vnpv = vnpvHC)

##################################################### Variable Coefficients #####################################

# Model 4.1: Variable coefficients, China primary refineries
vc_cn_pri <- pvcm(log(CU) ~ log(TCRC)+ log(SP2), 
                  data=p_cn_primary, model='random')
summary(vc_cn_pri, vnpv = vnpvHC)

# Model 4.2: Variable coefficients, RoW primary refineries
vc_rw_pri <- pvcm(log(CU) ~ log(TCRC)+ log(SP2),
                  data=p_rw_primary, model='random')#, na.action=na.exclude)
summary(vc_rw_pri, vnpv = vnpvHC)

# Model 4.3: Variable coefficients, China secondary refineries
vc_cn_sec <- pvcm(log(CU) ~ log(TCRC) + log(SP2),
                  data=p_cn_secondary, model='random')
summary(vc_cn_sec, vnpv = vnpvHC)

# Model 4.4: Variable coefficients, RoW secondary refineries
vc_rw_sec <- pvcm(log(CU) ~ log(TCRC) + log(SP2), 
                  data=p_rw_secondary, model='random', effect='individual')#, na.action=na.exclude)
summary(vc_rw_sec, vnpv = vnpvHC)

# Model 4.5: Variable coefficients, China all refineries
vc_cn_tot <- pvcm(log(CU) ~ log(TCRC) + log(SP2), 
                  data=p_cn_total, model='random', effect='individual')
summary(vc_cn_tot, vnpv = vnpvHC)

# Model 4.6: Variable coefficients, RoW all refineries
vc_rw_tot <- pvcm(log(CU) ~ log(TCRC), 
                  data=p_rw_total, model='random', effect='individual')#, na.action=na.exclude)
summary(vc_rw_tot, vnpv = vnpvHC)

##################################################### Dynamic models #####################################################



# Model 4.1: Dynamic models, no primary
dnm_cn_primary <- pgmm(log(CU) ~ lag(log(CU),1) + lag(log(TCRC),0)
               | lag(log(TCRC),2:99), data = p_cn_primary, 
               effect = "individual", model = "twosteps")
summary(dnm_cn_primary, robust = T, vnpv = vnpvHC)

# Model 4.2: Dynamic models, only primary
dnm_p <- pgmm((RR) ~ lag((RR),1) + lag((TCM.C),0)
              | lag((TCM.C),2:99), data = p_p, 
              effect = "individual", model = "twosteps")
summary(dnm_p, robust = T, vnpv = vnpvHC)

# Model 4.3: Dynamic models, both
dnm_both <- pgmm(log(RR) ~ lag(log(RR),1) + lag(log(TCM.C),0)
                 | lag(log(TCM.C),2:99), data = p_both, 
                 effect = "individual", model = "twosteps")
summary(dnm_both, robust = T, vnpv = vnpvHC)

## Tests
# Test 1: Unobserved effects
pwtest(log(RR) ~ log(TCM.C), data=p_np)
# npnclusion: unobserved effects exist, reject pooled OLS

# Test 2: Individual and time effects
plmtest(log(RR) ~ log(TCM.C), data=p_np, 
        effect = "individual", type = "honda")
plmtest(log(RR) ~ log(TCM.C), data=p_np,
        effect = "time", type = "kw")
# npnclusion: individual effect yes, time effect no

# Test 3: Hausman test, fixed vs random
w <- plm(log(RR) ~ log(TCM.C), data=p_np, model = "within")
r <- plm(log(RR) ~ log(TCM.C), data=p_np, model = "random")
phtest(w, r)
# npnclusion: RE is innpnsistent, FE is npnsistent

# Test 4: FE vs FD
pwfdtest(log(RR) ~ log(TCM.C), data=p_np, h0='fd')
pwfdtest(log(RR) ~ log(TCM.C), data=p_np, h0='fe')
# Both models have serial correlation

# Test 5: Serial correlation of FE
pbgtest(fe_np, order = 10)
# npnclusion: Serial correlation in original errors


npeftest(re_ori, vnpv = pvnpvHC)
