# Phononic-eigenvalue-problem

The latest version of code are ALL in /notebooks and their corresponding python code in /python_code

The data have to be download using wget.sh: create a folder named /data and move wget.sh into it and then ./wget.sh

To train and evaluate models, first create a folder named /model

# About code
notebooks/analyse_Model : to load, summary and evaluate a particular model with test set.

notebooks/inverse_Problem : To approximate the inverse problem by looking for the largest band-width gap between e2 and e3 and its corresponding input.

notebooks/NN_completeSet-mulGPU : To construct a forward CNN and train it with the dataset which is .h5 file

notebooks/InverseNN_M_n_e_to_pred_Q : This is just a little try on predicting wave numbers with material filed and eigenvalues and is not yet trained and tested due to limited time.
