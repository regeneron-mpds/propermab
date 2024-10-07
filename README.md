# propermab

`propermab` is a Python package for calculating and predicting molecular features and properties of monoclonal antibodies (mAbs).


## Installation (Linux)

First set up a conda environment by running the following commands on the terminal
```bash
git clone https://github.com/regeneron-mpds/propermab.git
conda env create -f propermab/conda_env.yml
conda activate propermab
```
Now install the `propermab` package with
```bash
pip install -e propermab/
```

### APBS
The APBS tool v3.0.0 is used by `propermab` to calculate electrostatic potentials. Download the tool and unzip it to a directory of your choice.
```bash
wget https://github.com/Electrostatics/apbs/releases/download/v3.0.0/APBS-3.0.0_Linux.zip -O apbs.zip
unzip apbs.zip
```
Record the path to this directory as it will be used in the next step. 

### Configuration
Edit the `default_config.json` file to specify the path for each of the entries in the file
```python
{
    "hmmer_binary_path" : "",
    "nanoshaper_binary_path" : "/ABPS_PATH/APBS-3.0.0.Linux/bin/NanoShaper",
    "apbs_binary_path" : "/ABPS_PATH/APBS-3.0.0.Linux/bin/apbs",
    "pdb2pqr_path" : "pdb2pqr",
    "multivalue_binary_path" : "/ABPS_PATH/APBS-3.0.0.Linux/share/apbs/tools/bin/multivalue",
    "atom_radii_file" : "",
    "apbs_ld_library_paths" : ["LIB_PATH", "/ABPS_PATH/APBS-3.0.0.Linux/lib/"]
}
```

You can find the value of `hmmer_binary_path` by issuing the following command on your terminal
```bash
dirname $(which hmmscan)
```

The value of `atom_radii_file` should point to a file named `amber.siz`. This file is part of the 
dependencies to run NanoShaper and can be obtained from NanoShaper repository (https://gitlab.iit.it/SDecherchi/nanoshaper).

To get the value for LIB_PATH, first create a separate conda environment to install the `readline 7.0` package.
```bash
conda deactivate
conda env create --name readline python=3.8
conda install readline=7.0
```
This may sound a bit involved, but it is necessary as the APBS tool specifically requires the readline.so.7 library file. `readline 7.0` can't be installed from within the propermab conda environment because that would result in too many conflicts. With that being said, once the readline package is installed, the value for LIB_PATH can be found by
```bash
echo ${CONDA_PREFIX}/lib/
```
Finally, be sure to replace APBS_PATH with the actual path to the directory where the APBS tool was unzipped in the previous step.

Now deactivate the readline environment and reactivate the propermab environment.

## Example
### Using `propermab` Python API
You can calculate the molecular features directly from a structure PDB file. Note that this assumes that the residues in PDB file are IMGT numbered and that the heavy chain is named H and the light chain is named L.
```python
from propermab import defaults
from propermab.features import feature_utils

defaults.system_config.update_from_json('./default_config.json')

mol_feature = feature_utils.calculate_features_from_pdb('./examples/apbs/mAb1.pdb')
```
Or you can provide a pair of heavy and light chain sequences, `propermab` then calls the `ABodyBuilder2` model to predict the structure, which will be used as the input for feature calculation.
```python
from propermab import defaults
from propermab.features import feature_utils

defaults.system_config.update_from_json('./default_config.json')

heavy_seq = 'HEAVY_SEQ'
light_seq = 'LIGHT_SEQ'
mol_features = feature_utils.get_all_mol_features(heavy_seq, light_seq, num_runs=1)
```
Be sure to replace HEAVY_SEQ and LIGHT_SEQ with the actual sequences. Different runs of `ABodyBuilder2` can result in some difference in sidechain conformations due to the relaxation step in `ABodyBuilder2`. This in turn can affect values of some of the molecular features `propermab` calculates. If the average feature value across multiple runs is desired, one can increase `num_runs`. `get_all_mol_features()` returns a Python dictionary in which the keys are feature names and the values are the corresponding lists of feature values from multiple runs.

## Third-party software
`propermab` requires separate installation of third party software which may carry their own license requirements, and should be reviewed by the user prior to installation and use
