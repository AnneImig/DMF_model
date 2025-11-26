ptf $
TITLE  Fate of DMF Zhou, X., Jin, W., Sun, C., Gao, S.-H., Chen, C., Wang, Q., Han, S.-F., Tu, R., Latif, 
M. A., and Wang, Q.: Microbial degradation of N,N-dimethylformamide by Paracoccus sp. strain DMF-3 from activated 
sludge, Chemical Engineering Journal, 343, 324–330, https://doi.org/10.1016/j.cej.2018.03.023, 2018.


SOLUTION_MASTER_SPECIES
Dmf           Dmf              0       Dmf             73.09  
Dma           Dma              0       Dma             45
Mma           Mma              0       Mma             31
Form          Form             0       Form            45
Formaldehyde  Formaldehyde     0       Formaldehyde    30



SOLUTION_SPECIES
Dmf = Dmf
    log_k     0
Dma = Dma
    log_k     0
Mma = Mma 
    log_k     0
Form = Form
    log_k     0
Formaldehyde = Formaldehyde
    log_k     0


#---------------------------------------#
# ambient water in the experiemnt? 
#---------------------------------------#

SOLUTION 0
    temp      15
    pH        7.0
    units     mol/kgw
    density   1
     Dmf       400 mg/kgw 
    -water    1 # kg 

Incremental_Reactions True 

##################   KINETICS   #####################
KINETICS 


Dmf_Dma
    -formula  Dmf -1 Dma 1 H20 2 Form 1  #H+ 3 
    # K1 = mol / h  
     -parms   $DmfDma$  
     -steps   0 24*1      #24h 
Dma_Mma
    -formula Dma  -1 Mma  1 Formaldehyde 1
    # K2 = mol / day 
     -parms   $DmaMma$  

Mma_Ammonia
    -formula  Mma  -1 Formaldehyde 1 Amm 1
    #1 H+ 1
    # K2 = mol / day 
     -parms   $MmaNH3$  


RATES
##############################OC release##############################

Dmf_Dma
-start
01 mDmf       = tot("Dmf")
03 rate       = parm(1)*mDmf
10 moles      = rate *time
15 put(rate,01)
20 SAVE moles
-end

Dma_Mma
-start
01 mDma       = tot("Dma")
03 rate       = parm(1)*mDma
10 moles      = rate *time
15 put(rate,02)
20 SAVE moles
-end

Mma_Ammonia
-start
01 mMma       = tot("Mma")
03 rate       = parm(1)*mMma
10 moles      = rate *time
15 put(rate,03)
20 SAVE moles
-end
SELECTED_OUTPUT  1 # 1 by default
  -file output/DMF.sel
-reset false
-totals DMF

SELECTED_OUTPUT  2
  -file output/MMA.sel
-reset false
-totals MMA

SELECTED_OUTPUT  3
  -file output/DMA.sel
-reset false
-totals DMA

SELECTED_OUTPUT  4
  -file output/NH3.sel
-reset false
-molalities Amm

SELECTED_OUTPUT 5
   -file  output/Results.sel 
	-time true
	-reset false

USER_PUNCH 5

    -headings Time pH	pe	DMF DMA MMA Formaldehyde Formate Ammonia rate_Dmf_Dma rate_Dma_Mma rate_Mma_Ammonium
	-start

10 	PUNCH  time
20  PUNCH - la("H+") 
30  PUNCH - la("e-") 
40  PUNCH  tot("Dmf") tot("Dma")   tot("Mma")   tot("Formaldehyde")  tot("Form")  MOL('Amm')
50  PUNCH  get(01) get(02)  get(03) 
    -end
END