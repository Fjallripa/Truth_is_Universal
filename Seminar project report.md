*This is the final report for an implementation project of the seminar "Mechanistic Interpretability" in the winter term 2024/25 at Heidelberg University.* 

### What I did in this project
#### Background
The paper "Truth is Universal" worked on detecting lies produced by LLMs. It built upon previous work that had created linear classifiers trained on the internal activations of LLMs reading true and false statements. This paper discovered a 2D truth subspace in activation space and built a classifier narrowly improving on the previous state of the art.

#### Main theme of this project
I tried to build an even better lie detector on my own.

#### Details of what I did
1. got the repo up and running
	- got a compute cloud instance working to create the LLM activations needed for the paper's experiments.
	- recreated the paper's experimental results for three models
2. did some usability enhancements for the code
	- implemented a feature to save and load each experiment's results.
	- provided clearer documentation for others to reproduce the paper.
3. implemented and tested a new classifier
	- see [Discussion of results](Seminar%20project%20report.md#discussion-of-results) for a rough comparision with the others


### Experimental methodology
- Trying out a classifier or new models were the most feasible things I could accomplish in the time left after getting everything to working order.
	- I decided that I wanted to try out a classifier first.
- First, I considered what kind of classifier might add value (i.e. generalize better than the previous ones).
	- All classifiers in the paper are linear ones.
		- LR and CCS cover the whole activation space.
		- MM uses only $\textbf{t}_G$, the general truth direction.
		- TTPD uses $\textbf{t}_G$ and $\textbf{p}$, the polarity direction.
		- -> While one could to another linear classifier using the discovered 2D truth space ($\textbf{t}_G$ and $\textbf{t}_P$, the polarity-sensitive truth direction), thus would likely add almost no extra value compared with the existing ones.
	- So, could one create a non-linear classifier?
		- I came up with three architectures that would be easy to implement.
			- QDA
			- MLP
			- Decision Forest
		- There, my main worry was overfitting.
			- So I tried to keep the resulting classification boundary as simple as possible.
		- A closer look at the data in question releaved it to almost alway have an X-shape with some overlap between classes in the middle.
			- See the notebook `truth_directions.ipynb` for examples.
			- This means non-linear decision boundaries will have a difficult time beating linear ones and are likely more at risk of overfitting.
		- I ended up trying out a very simple MLP (3 layers, 11 neurons) with a roughly 3-piece linear decision boundary
			- See `MLP_classifier_prototyping.ipynb` for my initial test with the new classifier and `probes.py` for the eventual implementation.
			- The hope was that it would remain simple enough to generalize well and still beat out the linear classifiers if there is an advantage to be had with non-linearity on this activation data.
	- I included the MLP into the series of generalization experiments carried out in `lie_detection.ipynb`.
		- There, each classifier is trained and tested 20 times on seperate kinds of datasets (always domain generalization tests).
		- From that,  summary statistics are calculated, saved and plotted (the mean and its standard deviation of the classification accuracy across those 20 iterations).
		- For a summary of results, see `results_summary.ipynb`.


### Discussion of results
- My MLP classifier generalized slightly worse than the other models (except CCS) on average).
	- See `results_summary.ipynb` for details.
- Due to time running out, I haven't done a detailed investigation into why.
- My current hyporthesis is that the concern I had when choosing a classifier architecture, namely that fancier decision boundaries would lend themselves poorly for classifying the X-shaped data with significant overlap between labels in the center. Rather, they might just increase overfitting (see the [methodology section](Seminar%20project%20report.md#experimental-methodology) for a few more details).
- In retrospect, It would have been more interesting to try out the existing classifiers on new open-weight LLMs instead. Seeing how far they would generalize and whether they worked better or worse for larger LLMs, would have been very interesting. But oh well, time constraints.
