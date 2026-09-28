Soboloev-Rollout Late Pleistocene Neural Emulator (SR-LPNE) 
-----------------------------------------------------------

This folder contains all the important files for the SR-LPNE surrogate model and its benchmarking using the Saltzman-Maasch 1990 model (SM90) with its 
specific Late Pleistocene solution. 

Source: 
https://www.cambridge.org/core/journals/earth-and-environmental-science-transactions-of-royal-society-of-edinburgh/article/abs/firstorder-global-model-of-late-cenozoic-climatic-change/57EF47DC4AA1B6A8A7D883C8368FD85D?utm_campaign=shareaholic&utm_medium=copy_link&utm_source=bookmark

File Structure and Content
--------------------------
- Data
    - SM90 
        - SM90_Insol.csv [normalized insolation used as forcing for the SM90 model]
        - SM90_X.csv     [SM90 model output for X = non-dimensional ice volume]
        - SM90_Y.csv     [SM90 model output for Y = non-dimensional CO_2 concentration]
        - SM90_Z.csv     [SM90 model output for Z = non-dimensional deep ocean temperature]

    - Observations
        - EDC_dust_on_GICC05.xls [EPICA Dome C Dust record]
        - edc-ch4-2008-noaa.txt  [EPICA Dome C CH_4 record]
        - edc3-composite-co2-2008-noaa_Vodstock.txt [EPICA Dome C CO_2 record]
        - EPICA_Dome_C_d18O.tab  [EPICA Dome C \delta^18 O record]
        - huybers06_65north.csv [Integrated summer insolation at 65 °N]


Saltzman_Model.ipynb:

generates a time series using the Late Pleistocene solution of the SM90 (content of ./Data/SM90) for a chosen time range forced by the insolation in ./Data/Observations/huybers06_65north.csv

Train_Emulator_SR-LPNE.py 

python file to train and test the emulator against the SM90 model data with custom rollout training loop and C_2 constraint based callback 