###### Contents:
1. [Setting up the repository](Step-by-step%20guide%20to%20reproduce%20the%20main%20results#Setting%20up%20the%20repository)
2. [Running the experiments](Step-by-step%20guide%20to%20reproduce%20the%20main%20results#Running%20the%20experiments)



#### Setting up the repository
*Step-by-step guide to get the "Fjallripa/Truth_is_Universal" repository up-and-running on a new computer.*

**!Note:** Instructions on how to install and run the LLMs are **not included** here!
For this minimal reproduction, only the relevant pre-computed LLM activation vectors will be provided as a download ([See below.](Step-by-step%20guide%20to%20reproduce%20the%20main%20results#3.%20Download%20the%20activations) The link is open until 2025-06-01 00:00).

###### 0. Hardware & software requirements
- This setup guide assumes you're working with a **bash-like shell** in a Unix-like environment (includes macOS) and are familiar with using it.
	- Also, you need to download a couple of GB of data, so a reasonable internet connection is required too.
- You also need **10 GB** of harddrive **space** (mainly for the activation vectors).
- Running `lie_detection.ipynb` in Jupyter Lab took up to about **5 GB** of **RAM** for me.
	- So even a PC with 8 GB of RAM should be able to handle this.
- No GPU required.


###### 1. Create conda environment for the repo
1. If needed, install conda first.
```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
```

2. Create the environment.
	- Packages will be installed later with pip.
```bash
conda create --name tiu python=3.11
conda activate tiu   # tiu = truth_is_universal
```

###### 2. Download the repository
1. Go to github.com/Fjallripa/Truth_is_Universal
2. Click the button "Code" on the right side. Click "Download ZIP".
3. Back in the shell, unzip and move the repo to your preferred location.
```bash
cd ~/Downloads
unzip Truth_is_Universal-main.zip
mv Truth_is_Universal-main Truth_is_Universal   # just a rename for convenience
mv Truth_is_Universal <dir>   # <dir> = the folder under which you want to store the repo
```

###### 3. Download the activations
1. Go to [heiBox](https://heibox.uni-heidelberg.de/d/4c33de9e1273401088de/) and download the `acts.zip` file (4.7 GB).
	+ The link is open until 2025-06-01 00:00.
2. Unzip it and move the `acts/` folder into the repo folder `Truth_is_Universal/`.
```bash
unzip acts.zip
mv acts <dir>/Truth_is_Universal
```

###### 4. Install the rest of the environment
1. Install all the python packages needed for this repo to the `tiu` conda environment.
```bash
cd <dir>/Truth_is_Universal
pip install -r ./requirements.txt
```

###### 5. Make the notebooks work
- The Jupyter notebooks (where the main experiments are) need to use the Python kernel from the `tiu` environment in order to work properly. 
	- If you're using VSCode, the `tiu` kernel should appear as an option when you're starting to run a notebook.
1. If you're using Jupyter Notebook or Lab, you need to enable that kernel as an option first:
```bash
ipython kernel install --user --name=tiu
```


=> With that, the repository set-up is finished!



#### Running the experiments
*Guide to do a minimal reproduction of the experiments.*

There are three notebooks producing the relevant experimental results.
- `./truth_directions.ipynb` - creates the plots showing the 2D truth space and the PCA analysis
	- On my PC, it took roughly 1 min to run for each model.
- `./lie_detection.ipynb` - trains different truth/falsehood classifiers and tests how well they generalize to new domains.
	- It takes roughly 25 mins to run for each model.
- `./result_summary.ipynb` - produces plots summarizing the results from `lie_detection.ipynb`.
	- It runs instantly, plotting the saved result data for all models at once.

Each one can be run from the get-go
0. Open the notebook
1. Choose the kernel 'tiu'
2. Use the Run-All command.
- The plots and summary data from each experiment are saved automatically under `./results/`.
* The only setting you may want to change is the model info if you want to try it for another model than Llama3_8B_chat. 
	- see the instructions at the top of the second cell "Hyperparameters".