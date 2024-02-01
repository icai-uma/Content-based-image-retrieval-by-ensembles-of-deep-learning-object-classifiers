In order to run the code, please change parentDir variable in train.py,data_split_caltech.py, data_split_imagenet.py, and paths.py files to the full path of the parent directory. Also, make sure to install split_folders before you split the datasets. 

Files:
------
main.py: the main file, where the precision, recall and mean average precision are computed.

run.py: runs the main file.

train.py: Used to train caltech256 dataset models.

caltech_prepare_models.py/imagenet_prepare_models.py: Used to compile the trained models, get their predictions, and create the ensemble.

data_split_caltech.py: Used to split the Caltech256 dataset into train, test, validations sets, and create bags.

data_split_imagenet.py: Used to split imagenet dataset into 1000 categories. After that the dataset is split into retrieval and test sets.

paths.py: Contains necessary paths to run other files.

Folders:
--------
caltech20, caltech50, imagenet: Contains images datasets, and all necessary files to run experiments on each dataset:

	checkpoints: The location where the weights of the trained models 		are saved during the training phase.

	trained models: The final weights corresponding to the models with 	the best accuracy.

	compiled models: contains the compiled bagged models.

	images: contains the downloaded dataset before 	splitting.

	splitted dataset: contains the bags of the dataset, test, 			validation, and base training  sets.

	vectors probabilities: contains the predicted probabilities of 			each trained model on each dataset for training and testing 			samples. 
		base: Contains models predicted probabilities.
		baseBest: Contains base models sorted from best to worst 			in terms of accuracy.
		baseWorst: Contains base models sorted from worst to best 			in terms of accuracy.
		ensemble: Containes all ensembles (from 2 to max ensemble 			size) predicted probabilities.
        	ensembleBest: Contains all ensembles sorted from best to 			worst.
		ensembleWorst: Contains all ensembles sorted from worst to 		best.
		mean: Contains the mean ensembles.
		median: Contains the median ensembles.
		query expansion: Contains the mean/median ensembles for 			testing set after performing the query expansion process.

