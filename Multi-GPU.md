Notes to Make it Work with Multiple GPUs
===

__Install__


Step 1: Install OpenMPI

First following the steps [here](https://www.open-mpi.org/faq/?category=building#easy-build).

Then add the following to .bashrc

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
```

Step 2: Install NCCL2 (Now for CUDA9. Will be updated for newer version.)

Download [NCCL2](https://developer.nvidia.com/compute/machine-learning/nccl/secure/v2.3/prod2/CUDA9.0/txz/nccl_2.3.5-2-cuda9.0_x86_64)

```
sudo scp lib/libnccl* /usr/lib/x86_64-linux-gnu/
```

```
sudo scp include/nccl.h  /usr/include/
```

Download [this](https://developer.nvidia.com/compute/machine-learning/nccl/secure/v2.3/prod2/nccl-repo-ubuntu1604-2.3.5-ga-cuda9.0_1-1_amd64)

```
sudo apt install ./nccl-repo-ubuntu1604-2.3.5-ga-cuda9.0_1-1_amd64.deb
```

Step 3: Install Horovod

```
HOROVOD_GPU_ALLREDUCE=NCCL
HOROVOD_NCCL_HOME=/usr/lib/x86_64-linux-gnu/
HOROVOD_NCCL_INCLUDE=/usr/include/ HOROVOD_WITHOUT_PYTORCH=1 pip install --no-cache-dir horovod
```

__Data__

Download data and pre-trained models as described in the original readme.

Step environmental variables:
```
export SQUAD_DIR=/home/ubuntu/git/bert/squad1.1
export BERT_BASE_DIR=/home/ubuntu/demo/model/uncased_L-12_H-768_A-12
```


__Run__

Train
```
mpirun -np 2     -H localhost:2     -bind-to none -map-by slot     -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH     -mca pml ob1 -mca btl ^openib     python run_squad_hvd.py   --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt   --do_train=True   --train_file=$SQUAD_DIR/train-v1.1.json   --do_predict=True   --predict_file=$SQUAD_DIR/dev-v1.1.json   --train_batch_size=12   --learning_rate=3e-5   --num_train_epochs=2.0   --max_seq_length=256   --doc_stride=128   --output_dir=/tmp/squad_base/
```

Evaluate
```
python $SQUAD_DIR/evaluate-v1.1.py $SQUAD_DIR/dev-v1.1.json /tmp/squad_base/predictions.json
```