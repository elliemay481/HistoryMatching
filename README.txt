HistoryMatching is a Python package used to history match a computer model. The aim is to iteratively identify non-implausible regions of a multi-dimensional parameter space.



To run a history match, the following procedure is:

    1. Initialise the history matching class, specifying the dimensionality of parameter space and choice of nonimplausible volume.

    2. Initilise the results class to store history match results.

    3. Define the computational model. The model takes the form of a python function, with an array of input parameters as the first function argument. The following arguments may be other model variables. The output of the function must be a single float.

    4. Define the parameter space to be searched, in terms of minimum and maximum values for each parameter.

    5. Set the data to be used. Each data point corresponds to a model output, and thus a corresponding array containing model variables used to produce this output is required. Uncertainties must also be defined here, corresponding to each data point.

    6. Run the history match. The entire history match may be run at once, with the number of waves specified. This will used the data defined in the previous step.
        - Alternatively, each wave can be run separately by defining the observational data to be used in this wave. This allows for a better exploration of the effect of including certain observables.

    7. Results are stored to a pickle file. They can be accessed by calling, e.g., Results.nonimplausible, where a list is returned. The length of the list is the number of history matching waves performed. Each element of the list is an array of nonimplausible samples returned at the end of the respective wave. Note that the final 3 columns in the array correspond to a sample's implausibility values.




Examples of history matches may be found in:
    - LiquidDropModel :     A history match is performed on the nuclear liquid drop model.
                            Required data file: 'MassEval2016.dat'
    - ChiralEFT :           A history match is performed on a Chiral Effective Field theory model
                            using scattering phase shifts and scattering observables.
                            Required data files: 'granada_phaseshift_highE.csv', 'Observables_highE.csv' 
                            Note: To run this, the nn_mwpc code is required.


Library requirements:

numpy
matplotlib
scipy
pickle
tqdm
abc
GPy
pyDOE

