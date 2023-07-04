**ThermoPalm**
===============

ThermoPalm is a tool to predict protein thermostability using protein language embeddings.
Embeddings are retrieved from ProtT5-XL-Uniref50 and a ridge regression model, fitted with
melting temperature values of the Meltome Atlas dataset, is applied to predict the melting
temperatures of specified proteins.


Usage 
-------------

.. code:: shell-session

    git clone https://github.com/jafetgado/ThermoPalm.git
    cd ThermoPalm
    conda env create -f ./env.yml -p ./env
    conda activate ./env
    python ./thermopalm/predict.py \
        --fasta_path "path_to_seq_file.fasta" \
        --save_dir ./output_dir \
        --csv_name predictions.csv \
        --verbose 1 


Citation
----------
If you find ThermoPalm useful, please cite the following:

Gado JE et al, 2023. Language model embeddings excel at predicting protein thermostability.
