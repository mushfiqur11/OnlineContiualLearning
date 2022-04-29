NUM_RUNS=1
BATCH_SIZE=10
LOG_DIR='logs/'

if [ ! -d $LOG_DIR ]; then
    mkdir -pv $LOG_DIR
fi

#comparation with fixed capacity methods

#permuted MNIST
python -u run.py --dataset p-MNIST --arch mlp --method RLL --num-runs 5 --num-task 20 --epoch 1 --maxstep 1000 --batch-size $BATCH_SIZE --optim SGD --learning-rate 0.1  --log-dir $LOG_DIR
python -u run.py --dataset p-MNIST --arch mlp --method SGD --num-runs 5 --num-task 20 --epoch 1 --maxstep 1000 --batch-size $BATCH_SIZE --optim SGD --learning-rate 0.1  --log-dir $LOG_DIR
python -u run.py --dataset p-MNIST --arch mlp --method STL --num-runs 5 --num-task 20 --epoch 1 --maxstep 1000 --batch-size $BATCH_SIZE --optim SGD --learning-rate 0.1  --log-dir $LOG_DIR

#rotated MNIST
python -u run.py --dataset r-MNIST --arch mlp --method RLL --num-runs 5 --num-task 20 --epoch 1 --maxstep 1000 --batch-size $BATCH_SIZE --optim SGD --learning-rate 0.1  --log-dir $LOG_DIR
python -u run.py --dataset r-MNIST --arch mlp --method SGD --num-runs 5 --num-task 20 --epoch 1 --maxstep 1000 --batch-size $BATCH_SIZE --optim SGD --learning-rate 0.1  --log-dir $LOG_DIR
python -u run.py --dataset r-MNIST --arch mlp --method STL --num-runs 5 --num-task 20 --epoch 1 --maxstep 1000 --batch-size $BATCH_SIZE --optim SGD --learning-rate 0.1  --log-dir $LOG_DIR

#20-split-CIFAR
python -u run.py --method RLL --num-runs 5 --num-task 20 --epoch 8 --batch-size $BATCH_SIZE --optim SGD --learning-rate 0.03  --log-dir $LOG_DIR
python -u run.py --method STL --num-runs 5 --num-task 20 --epoch 8 --batch-size $BATCH_SIZE --optim SGD --learning-rate 0.03  --log-dir $LOG_DIR
python -u run.py --method SGD --num-runs 5 --num-task 20 --epoch 8 --batch-size $BATCH_SIZE --optim SGD --learning-rate 0.03  --log-dir $LOG_DIR

#20-split-miniImageNet
python -u run.py --dataset miniImageNet --method RLL --num-runs 5 --num-task 20 --epoch 8 --batch-size $BATCH_SIZE --optim SGD --learning-rate 0.03  --log-dir $LOG_DIR
python -u run.py --dataset miniImageNet --method STL --num-runs 5 --num-task 20 --epoch 8 --batch-size $BATCH_SIZE --optim SGD --learning-rate 0.03  --log-dir $LOG_DIR
python -u run.py --dataset miniImageNet --method SGD --num-runs 5 --num-task 20 --epoch 8 --batch-size $BATCH_SIZE --optim SGD --learning-rate 0.03  --log-dir $LOG_DIR


#experiment with more archs

#CIFAR
python -u run.py --arch lenet --method RLL --num-runs 5 --num-task 20 --epoch 20 --batch-size $BATCH_SIZE --optim SGD --learning-rate 0.03  --log-dir $LOG_DIR --gpuid 0
python -u run.py --arch lenet --method STL --num-runs 5 --num-task 20 --epoch 20 --batch-size $BATCH_SIZE --optim SGD --learning-rate 0.03  --log-dir $LOG_DIR --gpuid 0

python -u run.py --arch alexnet --method RLL --num-runs 5 --num-task 20 --epoch 20 --batch-size $BATCH_SIZE --optim SGD --learning-rate 0.03  --log-dir $LOG_DIR --gpuid 0
python -u run.py --arch alexnet --method STL --num-runs 5 --num-task 20 --epoch 20 --batch-size $BATCH_SIZE --optim SGD --learning-rate 0.03  --log-dir $LOG_DIR --gpuid 0

python -u run.py --arch vgg11 --method RLL --num-runs 5 --num-task 20 --epoch 20 --batch-size $BATCH_SIZE --optim SGD --learning-rate 0.03  --log-dir $LOG_DIR --gpuid 0
python -u run.py --arch vgg11 --method STL --num-runs 5 --num-task 20 --epoch 20 --batch-size $BATCH_SIZE --optim SGD --learning-rate 0.03  --log-dir $LOG_DIR --gpuid 0

python -u run.py --arch vgg13 --method RLL --num-runs 5 --num-task 20 --epoch 20 --batch-size $BATCH_SIZE --optim SGD --learning-rate 0.03  --log-dir $LOG_DIR --gpuid 0
python -u run.py --arch vgg13 --method STL --num-runs 5 --num-task 20 --epoch 20 --batch-size $BATCH_SIZE --optim SGD --learning-rate 0.03  --log-dir $LOG_DIR --gpuid 0

#miniImageNet
python -u run.py --dataset miniImageNet --arch alexnet2 --method RLL --num-runs 5 --num-task 20 --epoch 20 --batch-size $BATCH_SIZE --optim SGD --learning-rate 0.01  --log-dir $LOG_DIR --gpuid 0
python -u run.py --dataset miniImageNet --arch alexnet2 --method STL --num-runs 5 --num-task 20 --epoch 20 --batch-size $BATCH_SIZE --optim SGD --learning-rate 0.01  --log-dir $LOG_DIR --gpuid 0

python -u run.py --dataset miniImageNet --arch vgg11 --method RLL --num-runs 5 --num-task 20 --epoch 20 --batch-size $BATCH_SIZE --optim SGD --learning-rate 0.01  --log-dir $LOG_DIR --gpuid 0
python -u run.py --dataset miniImageNet --arch vgg11 --method STL --num-runs 5 --num-task 20 --epoch 20 --batch-size $BATCH_SIZE --optim SGD --learning-rate 0.01  --log-dir $LOG_DIR --gpuid 0

python -u run.py --dataset miniImageNet --arch vgg13 --method RLL --num-runs 5 --num-task 20 --epoch 20 --batch-size $BATCH_SIZE --optim SGD --learning-rate 0.01  --log-dir $LOG_DIR --gpuid 0
python -u run.py --dataset miniImageNet --arch vgg13 --method STL --num-runs 5 --num-task 20 --epoch 20 --batch-size $BATCH_SIZE --optim SGD --learning-rate 0.01  --log-dir $LOG_DIR --gpuid 0

python -u run.py --dataset miniImageNet --arch resnet18 --method RLL --num-runs 5 --num-task 20 --epoch 20 --batch-size $BATCH_SIZE --optim SGD --learning-rate 0.01  --log-dir $LOG_DIR --gpuid 0
python -u run.py --dataset miniImageNet --arch resnet18 --method STL --num-runs 5 --num-task 20 --epoch 20 --batch-size $BATCH_SIZE --optim SGD --learning-rate 0.01  --log-dir $LOG_DIR --gpuid 0

