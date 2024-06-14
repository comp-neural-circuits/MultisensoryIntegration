# Multimodal

Parameters are stored in parameters.txt

batch_simulate_monte_carlo.py
    Runs a batch of simulations
    In this file you can specify if any values are pertubed in simulations under the function worker.
        The following code is included:
        For Figure (3, 5) correlation is chosen from a random values between 0 and 1 (Line 54).
        For Figure (4) S1 has a greater initial bias (This is commented out. Lines 58-64)

    It required the following files to run the batch:
        parameters.txt
        tools/biased_weights.py - sets up the initial weight matrix in V1 and S1
        tools/runRimulation.py - runs the actual simulation and updates the weight matrix if hebbian plasticity is ON ("Hebbflag":1)

        analysis/making_figures.py created the plots from a simulation for the 
        'sample2_RF.pdf' - shows the final RF for both V1 and S1
        'samples_v1.pdf' - is the evolution of weights over the simulation for V1
        'samples_s1.pdf' - is the evolution of weights over the simulation for S1

        analysis/making_stats.py
        proportions.pdf - plots how many neurons are unimodal in V1, S1, or bimodal
        hist.pdf - plots a histogram of Receptive field sizes
        stats.txt - hold the values for theta, correlation, receptive field size V1, receptive field size V1, 
            number of unimodal neurons in V1, number of unimodal neurons in S1, number of bimodal neurons 

To run one simulation run runOneSim.py (uses same files as batch_simulate_monte_carlo)

batch_analyze_monte_carlo_varying_correlation.py
    Figure (3) with varying correlation values over simulations
        takes all batch simulations and calculated figures for manuscript.
        calculates the initial weight matrix and the final weight template
        calculates correlation vs togpography/alighment/bimodal cells
        figures are not polished

batch_analyze_monte_carlo_varying_S1bias.py
    Figure (4) with varying S1 bias over simulations
        calculated S1 vs V1 topography along increasing S1 bias - used for 
        figures are not polished

batch_analyze_recontruction.py
    Figure (5) with reconstruction or Rsquared 
    Uses the final weight matrix to get activity for the correlation from figure (3) with varying correlations
    fits one linear model between RL(x) and V1(y) and S1(y)
        uses this modesl to calculate the R^2 (how much variance can be explained) from V1 or S1 individually
        plots the figures for  fig 6, unpolished

linear_mixed_model_analysis.R code is written in R
    Figure (1) - caculates several linear models to determine if age and area has an effect on various characterisitcs in the regions, 
    i.e. amplitude of calcium events, duration of events, participation rate, and event rates = 1/(IEI/7.7ms) (frame rate)
    Calculates the effect sizes of each of these
    Participation rate uses a logit link function because these values are proportions



 
